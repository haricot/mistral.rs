import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qualified_r1_publication', ROOT / 'promote-qualified-r1.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublicationTests(unittest.TestCase):
    def test_exact_successful_run_and_all_hosted_gpu_gates(self):
        receipt = {'run_id':1, 'run_attempt':1, 'source_sha':'a'*40, 'candidate':'b'*40,
                   'gpu_job_id':9, 'artifacts':[{'id':3,'name':'proof-1-1','digest':'sha256:'+'c'*64}]}
        source = {'id':1,'run_attempt':1,'head_sha':'a'*40,'status':'completed','conclusion':'success',
                  'event':'push','head_repository':{'full_name':module.REPOSITORY}}
        jobs = {'jobs':[{'id':i,'name':n,'run_attempt':1,'conclusion':'success'} for i,n in enumerate(module.GATES)]}
        jobs['jobs'].append({'id':9,'name':'SM61 Mistral Extended runtime (candidate='+'b'*40+')',
                             'run_attempt':1,'conclusion':'success','labels':['mistral-extended','mistral-extended-1']})
        jobs['total_count'] = len(jobs['jobs'])
        artifacts = {'total_count':1,'artifacts':[{**receipt['artifacts'][0],'expired':False}]}
        module.verify_run(receipt,source,jobs,artifacts)
        for key,value in (('head_sha','d'*40),('run_attempt',2),('conclusion','failure')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.verify_run(receipt,{**source,key:value},jobs,artifacts)
        for index in range(len(jobs['jobs'])):
            wrong = copy.deepcopy(jobs)
            wrong['jobs'][index]['conclusion']='failure'
            with self.subTest(job=index), self.assertRaises(ValueError):
                module.verify_run(receipt,source,wrong,artifacts)
        bad = copy.deepcopy(artifacts)
        bad['artifacts'][0]['digest']='sha256:'+'d'*64
        with self.assertRaises(ValueError):module.verify_run(receipt,source,jobs,bad)

    def test_verified_zip_rejects_modified_bytes_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            archive=root/'artifact.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('report.json','{}')
            module.unpack(archive,root/'output',module.sha(archive))
            self.assertEqual((root/'output/report.json').read_text(),'{}')
            with self.assertRaises(ValueError):module.unpack(archive,root/'output','0'*64)
            with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside','no')
            with self.assertRaises(ValueError):module.unpack(archive,root/'output',module.sha(archive))

    def test_atomic_source_leases_and_same_sha_idempotence(self):
        plan={'candidate':'e'*40,'components':{'legacy':'a'*40,'extended':'e'*40},
              'config':{'components':{'legacy':{'branch':'allow_old_card'}},
                        'targets':{'extended':{'branch':'extended'}}},'target':'extended',
              'leases':{'allow_old_card':'b'*40,'extended':'c'*40,'master':'d'*40}}
        current=dict(plan['leases'])
        args=module.push_arguments('/repo',plan,{'run_id':1},current)
        self.assertIn('--atomic',args)
        self.assertIn('--force-with-lease=refs/heads/extended:'+'c'*40,args)
        self.assertIn('c'*40+':refs/tags/archive/campaign/1/extended',args)
        current.update(module.branch_updates(plan))
        module.push_arguments('/repo',plan,{'run_id':1},current)
        current['master']='f'*40
        with self.assertRaises(ValueError):module.push_arguments('/repo',plan,{'run_id':1},current)

    def test_publication_policy_records_vl_gguf_without_changing_qualification(self):
        receipt=json.loads((ROOT.parent/'campaign/publish-r1.once.json').read_text())
        self.assertEqual(receipt['non_blocking_numeric_models'],['vl','gguf'])
        self.assertEqual(receipt['run_id'],41479428911)
        self.assertEqual(receipt['candidate'],'3966d0ad0ba5d1964e418a3c37d665c46a84a9aa')
        workflow=(ROOT.parent/'workflows/mistral-promote-qualified-r1.yml').read_text()
        self.assertNotIn('cargo build',workflow)
        self.assertNotIn('run-once.sh',workflow)
        self.assertNotIn('self-hosted',workflow)
        self.assertIn('queue: max',workflow)

    def test_numeric_vl_gguf_do_not_block_but_execution_integrity_and_text_still_do(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);directory=root/'release-r1';directory.mkdir()
            image=root/'image';image.mkdir();(image/'oci.json').write_text('{}')
            (root/'report.json').write_text('{}')
            policy={'models':{name:{'repo_id':name} for name in ('text','vl','gguf')}}
            (root/'models.json').write_text(json.dumps(policy));(root/'qualification.py').write_text('# pinned')
            measured={name:{'status':'PASS' if name=='text' else 'FAIL','cpu_executed':True,'cuda_executed':True}
                      for name in policy['models']}
            report={'schema':'haricot-extended-release-gpu-r1','release':{'binary_sha256':'a'*64},
                    'full_models_executed':True,'status':'COLLECTED','run_id':'1','run_attempt':'1',
                    'sm':'6.1','gpu_uuid':'GPU-fixture','driver':'fixture','models':{},'files':{},
                    'kernel_report_sha256':module.sha(root/'report.json'),
                    'image_proof_sha256':module.sha(image/'oci.json'),
                    'policy_sha256':module.sha(root/'models.json'),
                    'harness_sha256':module.sha(root/'qualification.py'),
                    'paged_fp8':{'status':'PASS','errors':{},'binary_sha256':'a'*64}}
            for name in measured:
                report['models'][name]={**measured[name],'weights':policy['models'][name],'errors':{}}
                for device in ('cpu','cuda'):
                    path=directory/(name+'-'+device+'.json');path.write_text(json.dumps({'mode':name}))
                    report['files'][path.name]=module.sha(path)
            qualification=SimpleNamespace(ROOT=root,measure=lambda c,g,p:measured[c['mode']])
            receipt={'run_id':1,'run_attempt':1,'non_blocking_numeric_models':['vl','gguf']}
            kernel={'run_id':'1','run_attempt':'1'}
            def verify(value):
                module.verify_models(value,kernel,report['release'],image,policy,qualification,receipt,directory)
            verify(report)
            bad=copy.deepcopy(report);bad['models']['vl']['errors']={'cuda':'execution failed'}
            with self.assertRaises(ValueError):verify(bad)
            measured['text']['status']='FAIL';bad=copy.deepcopy(report);bad['models']['text']['status']='FAIL'
            with self.assertRaises(ValueError):verify(bad)
            measured['text']['status']='PASS'
            (directory/'vl-cpu.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):verify(report)


if __name__=='__main__':unittest.main()
