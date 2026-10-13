#!/bin/zsh
cd "${0:A:h}/.." || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print '請先依 README 建立 Python 環境並安裝專案。'
  read '?按 Enter 結束…'
  exit 1
fi
.venv/bin/python -m race_gui.server
