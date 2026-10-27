import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mt.course_report import native_group_scores, verified_representation_cache


class CourseReportIntegrityTests(unittest.TestCase):
    def test_changed_completed_aggregate_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'report').mkdir()
            values={'metrics':{'generated_at_utc':'old','per_seed':{'42':{'complete':True}}},
                    'aggregate':{'n':3,'mean':20},'groups':{'test':[1,2]}}
            for key,value in values.items():(root/'report'/(key+'.json')).write_text(json.dumps(value))
            recomputed=copy.deepcopy(values);recomputed['metrics']['generated_at_utc']='new'
            with patch('mt.representation_report.collect',return_value=recomputed):
                verified_representation_cache(root)
                (root/'report/aggregate.json').write_text(json.dumps({'n':3,'mean':90}))
                with self.assertRaisesRegex(ValueError,'aggregate'):
                    verified_representation_cache(root)

    def test_native_groups_use_frozen_ids_and_reject_reference_damage(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'groups.json'
            text='Ein Satz.'
            examples=[{'id':4,'source':'A sentence.','reference':text,'prediction':text}]
            table={'definitions':{'source_words_boundaries':[10,20,40]},'rows':[
                {'id':4,'source_sha256':hashlib.sha256(b'A sentence.').hexdigest(),
                 'reference_sha256':hashlib.sha256(text.encode()).hexdigest(),'groups':{'source_words':0}}]}
            path.write_text(json.dumps(table))
            result=native_group_scores(path,examples)['source_words']
            self.assertEqual(result[0]['ids'],[4]);self.assertIn('tok:13a',result[0]['bleu_signature'])
            self.assertIsNone(result[1]['bleu']);self.assertIsNone(result[1]['chrf_signature'])
            examples[0]['reference']='changed'
            with self.assertRaisesRegex(ValueError,'identity'):
                native_group_scores(path,examples)


if __name__=='__main__':unittest.main()
