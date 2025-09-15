#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v python >/dev/null 2>&1; then
  python test.py | tee out.txt
else
  python3 test.py | tee out.txt
fi

grep -q "skills: AvoidSkill, CruiseSkill, GraspSkill\|skills: CruiseSkill, GraspSkill, AvoidSkill\|skills:" out.txt
grep -q "巡航：沿预设路径移动" out.txt
grep -q "抓取：完成目标抓取动作" out.txt
grep -q "规避：绕开障碍物" out.txt
echo "[OK] task5_skills 测试通过。"


