import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name('kmpc_log_digest.py')

class DigestTest(unittest.TestCase):
    def run_digest(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str,args)], capture_output=True, text=True)

    def test_events_extremes_truncation_and_window(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'kmpc_status.csv'
            with p.open('w', newline='') as f:
                w=csv.writer(f); w.writerow(['t_s','status'])
                for i in range(10000):
                    w.writerow([i, f'line={"stop" if i==50 else "on"} reason={"pitch" if i==50 else "none"} e0={42 if i==70 else 1}'])
            result=self.run_digest(p,'--limit','5')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('记录=10000',result.stdout)
            self.assertIn('reason:none→pitch',result.stdout)
            self.assertIn('最大=42(L72,t=70)',result.stdout)
            self.assertLess(len(result.stdout.splitlines()), 20)
            self.assertIn('pitch', self.run_digest('--window',f'{p}:52','--context','0').stdout)

    def test_waist_and_bad_row(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'servo.csv'
            p.write_text('t_s,seq,robot_id,joints_deg\n0,1,3,0 0 0\n1,2,3,0.1 0 0\n2,3,3,0 0 0\n3,4\n')
            r=self.run_digest(p)
            self.assertEqual(r.returncode,0)
            self.assertIn('疑似方向变化',r.stdout)
            self.assertIn('异常/残缺行=1',r.stdout)
            self.assertIn('腰J1指令度',r.stdout)

    def test_text_and_missing_input(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'node.log'; p.write_text('started\nERROR motor failure\nend\n')
            self.assertIn('关键词',self.run_digest(p).stdout)
            self.assertNotEqual(self.run_digest(Path(d)/'missing').returncode,0)

if __name__ == '__main__': unittest.main()
