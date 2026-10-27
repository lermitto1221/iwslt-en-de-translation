"""Freeze and run the paired BPE/unigram experiment with one shared schedule."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from .common import read_json, seed_all, sha256, write_json
from .data import read_pairs
from .explore import atomic_json, check_hashes, execution_lock
from .model import TranslationTransformer, parameter_counts
from .run_experiments import evaluate_or_reuse, frozen_protocol
from .summary import validate_saved_outputs
from .tokenizer import load
from .train import model_digest

DEFAULT_ROOT = "artifacts/representation_2026_10_27"
REPRESENTATIONS = ("bpe", "unigram")
SEEDS = (42, 43, 44)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def frozen_groups(root):
    """One representation-independent membership table per official split."""
    root = Path(root)
    design = read_json(root / "design.json")
    definitions = design["grouping"]
    counts_path = Path("artifacts/exploration_2026_10_27/data/word_frequency_en.jsonl")
    counts = {r["word"]:r["count"] for r in read_pairs(counts_path)}
    cluster_path = Path("artifacts/exploration_2026_10_27/vectors/assignments.jsonl")
    cluster = {r["id"]:r["cluster"] for r in read_pairs(cluster_path) if r["split"] == "validation"}
    tok = load("artifacts/data")
    if sha256("artifacts/data/spm.model") != definitions["anchor_tokenizer_sha256"]:
        raise ValueError("anchor tokenizer changed")
    result = {}
    for split in ("validation", "test"):
        rows = read_pairs(f"artifacts/data/{split}.jsonl")
        memberships = []
        for r in rows:
            words = [w.casefold() for w in r["en"].split()]
            features = {"source_words":len(words),
                        "source_rare_fraction":sum(counts.get(w,0)<=5 for w in words)/len(words),
                        "anchor_fragmentation":len(tok.encode(r["en"],out_type=int))/len(words)}
            groups = {key:sum(value > upper for upper in definitions[key+"_boundaries"])
                      for key,value in features.items()}
            if split == "validation":
                groups["cluster"] = cluster[r["id"]]
            memberships.append({"id":r["id"],"source_sha256":hashlib.sha256(r["en"].encode()).hexdigest(),
                                "reference_sha256":hashlib.sha256(r["de"].encode()).hexdigest(),
                                "features":features,"groups":groups})
        value = {"split":split,"n":len(rows),"source_file_sha256":sha256(f"artifacts/data/{split}.jsonl"),
                 "anchor_tokenizer_sha256":definitions["anchor_tokenizer_sha256"],
                 "training_word_frequency_sha256":sha256(counts_path),
                 "validation_assignments_sha256":sha256(cluster_path) if split=="validation" else None,
                 "definitions":definitions,"rows":memberships}
        path = root / "groups" / f"{split}.json"
        if path.exists() and read_json(path) != value:
            raise ValueError("frozen group membership differs; cannot redefine groups")
        if not path.exists():
            atomic_json(path,value)
        result[split] = {"file":str(path.resolve()),"sha256":sha256(path),"n":len(rows)}
    return result


def freeze(root=DEFAULT_ROOT):
    from .representation_train import execution_source_hashes
    from .representation_prepare import prepare
    root = Path(root)
    if (root / "plan.json").exists():
        raise FileExistsError("execution plan already exists; use run to verify/reuse it")
    prepared = prepare(root)
    if prepared.get("complete") is not True:
        raise ValueError("tokenizer and shared-schedule preparation is incomplete")
    for rel,digest in prepared["files"].items():
        if sha256(root/rel) != digest:
            raise ValueError("preparation evidence changed: " + rel)
    if any(any((root / "runs").glob(pattern)) for pattern in
           ("*/last.pt", "*/best.pt", "*/train.jsonl", "*/summary.json", "*/*/metrics.json")):
        raise ValueError("cannot freeze a plan after training started")
    design=read_json(root/"design.json")
    if (design["seeds"]!=list(SEEDS) or design["representations"]!=list(REPRESENTATIONS)
            or prepared["request"]["design_sha256"]!=sha256(root/"design.json")
            or sha256(design["decode"]["protocol"])!=design["decode"]["protocol_sha256"]):
        raise ValueError("preparation/design/shared-protocol identity differs")
    groups = frozen_groups(root)
    template = read_json("config.json")
    required={"max_updates":3000,"max_train_tokens":128,"batch_tokens":4096,"accum_steps":4,
              "validate_every":500,"label_smoothing":.1,"warmup_updates":300,"lr":.0005,
              "amp":True,"amp_dtype":"bfloat16","d_model":256,"nhead":4,"num_layers":3,
              "dim_ff":1024,"dropout":.1,"vocab_size":8000,"max_generation_tokens":256,
              "beam_size":4,"length_penalty":.6,"decode_batch_size":16,"deterministic":True}
    if any(template.get(k)!=v for k,v in required.items()):
        raise ValueError("training template differs from the fixed representation design")
    configs, initializations = {}, {}
    for seed in SEEDS:
        for rep in REPRESENTATIONS:
            name = f"{rep}_{seed}"
            cfg = dict(template, seed=seed, data_dir=(root/"data"/rep).as_posix(),run_dir=(root/"runs"/name).as_posix(),
                       representation=rep,batch_plan=(root/"batches"/f"seed{seed}.jsonl").as_posix(),
                       common_ids=(root/"common_ids.json").as_posix(),expected_validation_samples=888,expected_parameters=11681600)
            for key in ("batch_plan","common_ids"):
                cfg[key+"_sha256"] = sha256(cfg[key])
            if cfg["max_updates"]!=3000 or cfg["label_smoothing"]!=.1 or cfg["accum_steps"]!=4:
                raise ValueError("base configuration differs from approved fixed representation design")
            tok = load(cfg["data_dir"])
            if tok.vocab_size()!=8000:
                raise ValueError("both actual vocabularies must equal 8000")
            seed_all(seed)
            model = TranslationTransformer(tok.vocab_size(),cfg)
            params, initial = parameter_counts(model), model_digest(model)
            if params != {"total":11681600,"trainable":11681600}:
                raise ValueError("model parameter count differs from fixed design")
            if seed in initializations and initializations[seed]!=initial:
                raise ValueError("same seed initial numeric weights differ across representations")
            initializations[seed]=initial
            del model
            p=root/"configs"/(name+".json")
            if p.exists() and read_json(p)!=cfg:
                raise FileExistsError("refusing to overwrite a different configuration")
            if not p.exists():atomic_json(p,cfg)
            configs[name]={"file":str(p.resolve()),"sha256":sha256(p),"seed":seed,"representation":rep,"parameters":params}
    source_hashes=execution_source_hashes()
    source_hashes.update({name:sha256(Path(__file__).with_name(name)) for name in ("representation.py","representation_prepare.py","evaluate.py","inference.py","run_experiments.py","summary.py","explore.py")})
    frozen_inputs={str((root/rel).resolve()):digest for rel,digest in prepared["files"].items()}
    frozen_inputs.update({str((root/name).resolve()):sha256(root/name) for name in ("design.json","prepare_manifest.json")})
    frozen_inputs.update({info["file"]:info["sha256"] for info in configs.values()})
    frozen_inputs.update({info["file"]:info["sha256"] for info in groups.values()})
    protocol=Path("artifacts/decode_protocol_seed42.json")
    frozen_inputs[str(protocol.resolve())]=sha256(protocol)
    frozen_inputs[str(Path("config.json").resolve())]=sha256("config.json")
    frozen_protocol(protocol)
    record={"created_at_utc":now(),"design_sha256":sha256(root/"design.json"),"configs":configs,
            "initial_model_sha256":initializations,"source_hashes":source_hashes,"frozen_inputs":frozen_inputs,
            "groups":groups,"protocol":str(protocol.resolve()),"execution_order":list(configs),
            "budget":{"runs":6,"updates_per_run":3000},"selection":"highest validation greedy BLEU; earliest exact tie",
            "note":"New controlled representation experiment after previous results; old BPE is historical reference only. Same sentences/microbatch boundaries, representation-specific token counts."}
    snapshot=root/"source_snapshot"; snapshot.mkdir(exist_ok=True)
    for name,digest in source_hashes.items():
        content=Path(__file__).with_name(name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError("source changed during plan freeze")
        path=snapshot/name
        if path.exists() and sha256(path)!=digest:raise ValueError("source snapshot conflict")
        if not path.exists():path.write_bytes(content)
    # Publish the immutable completion marker only after every snapshot exists.
    with (root/"plan.json").open("x",encoding="utf-8") as handle:
        json.dump(record,handle,ensure_ascii=False,indent=2);handle.write("\n")
    print({"plan":str(root/"plan.json"),"common_training_pairs":prepared["common_training_pairs"],"runs":6},flush=True)
    return record


def validate_plan(root):
    root=Path(root); plan=read_json(root/"plan.json")
    check_hashes(plan["frozen_inputs"])
    for name,digest in plan["source_hashes"].items():
        if sha256(Path(__file__).with_name(name))!=digest or sha256(root/"source_snapshot"/name)!=digest:
            raise ValueError("execution source differs from frozen snapshot: "+name)
    return plan


def run(root=DEFAULT_ROOT):
    from .representation_train import train,validate_completed_run
    root=Path(root); plan=validate_plan(root); protocol=frozen_protocol(plan["protocol"])
    with execution_lock(root/"execution.lock"):
        for name in plan["execution_order"]:
            validate_plan(root)
            cfg=read_json(plan["configs"][name]["file"]); directory=Path(cfg["run_dir"])
            atomic_json(root/"progress.json",{"run":name,"stage":"training_or_verify","updated_at_utc":now()})
            print({"run":name,"stage":"training_or_verify"},flush=True)
            if (directory/"summary.json").is_file() and read_json(directory/"summary.json").get("complete"):
                summary=validate_completed_run(cfg)
            else:
                last=directory/"last.pt"
                summary=train(cfg,resume=last if last.exists() else None)
                validate_completed_run(cfg)
            link=read_json(directory/"data_link.json")
            if link["initial_model_sha256"]!=plan["initial_model_sha256"][str(cfg["seed"])]:
                raise ValueError("training initialization differs from pre-training audit")
            results={}
            for split in ("validation","test"):
                for method in ("greedy","beam"):
                    validate_plan(root)
                    key=split+"_"+method
                    atomic_json(root/"progress.json",{"run":name,"stage":key,"updated_at_utc":now()})
                    result=evaluate_or_reuse(directory/"best.pt",split,method,directory/key,
                        beam_size=protocol["beam_size"],alpha=protocol["length_penalty"],max_tokens=protocol["max_generation_tokens"])
                    validate_saved_outputs(result,directory/key,name+"/"+key)
                    results[key]=result
                    print({"run":name,"stage":key,"bleu":result["bleu"],"samples":result["samples"]},flush=True)
            done=root/"completed"/(name+".json")
            record={"run":name,"config_sha256":sha256(plan["configs"][name]["file"]),"training":summary,
                    "results":results,"initial_model_sha256":link["initial_model_sha256"]}
            if done.exists() and read_json(done)!=record:
                raise ValueError("completed run record differs from verified saved artifacts")
            if not done.exists():atomic_json(done,record)
            print({"run":name,"stage":"complete"},flush=True)
        validate_plan(root)
        atomic_json(root/"progress.json",{"stage":"complete","runs":plan["execution_order"],"updated_at_utc":now()})


def verify_history(root=DEFAULT_ROOT):
    root=Path(root); history=read_json(root/"historical_manifest.json")
    for name,entry in history["files"].items():
        if not Path(name).is_file() or sha256(name)!=entry["sha256"] or Path(name).stat().st_size!=entry["bytes"]:
            raise ValueError("historical file changed: "+name)
    record={"checked_at_utc":now(),"status":"verified","unchanged_files":len(history["files"])}
    atomic_json(root/"historical_verification.json",record)
    print(record,flush=True);return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("freeze","run","verify-history"))
    p.add_argument("--root",default=DEFAULT_ROOT)
    args=p.parse_args(); {"freeze":freeze,"run":run,"verify-history":verify_history}[args.command](args.root)


if __name__=="__main__":main()
