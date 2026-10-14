# Windows CI 來源驗證修正

2026-10-14 更新：commit `8d08d194` 的 Windows 來源驗證已成功，新的失敗發生在 unittest。使用者提供的完整日誌確認 `test_compact_and_full_recording_have_identical_training_and_feedback` 結束時，`history.sqlite` 被未關閉的連線占用，`TemporaryDirectory.cleanup()` 拋出 WinError 32；共 152 測試、只有此錯誤。這與下方舊版 CRLF 問題不同。

測試與 `suite_report.read_sessions` 均改用 `contextlib.closing`；SQLite 的 `with connection` 只提交／回滾交易，不會自動關閉連線。新增回歸檢查明確 close 後才刪除檔案。CI 啟用 UTF-8 並保存每個 runner 的完整測試日誌、短 suite 的執行日誌，避免下次只看到紅色摘要。新 commit 尚未推送，Windows 雲端結果仍待驗證，沒有宣稱「保證不會失敗」。

推送 commit `8c33e2b` 後，[Research verification](https://github.com/113ncuee-project/20261012_RACE-RL/actions/runs/38328098999) 的 Ubuntu／macOS 全部成功；Windows 在 `python tools/verify_sources.py` 失敗，後面的 RL／CNN 測試未執行。

原因由隔離 Git repository 重現：`core.autocrlf=true` 會將來源的 LF 簽出成 CRLF，4 個 RapidChiplet 固定來源檔案的原始位元組 SHA256 因此不符。加入 `.gitattributes` 的 `* text=auto eol=lf` 後，相同 Windows 簽出設定下，4 個原始位元組與鎖定雜湊全部符合。`.bat` 啟動器明確使用 CRLF。依據 [Git attributes 文件](https://git-scm.com/docs/gitattributes#_text)。

沒有修改第三方來源、鎖定雜湊、RL 算法或 PPA 分數，也沒有跳過 Windows 工作。`tests/test_checkout_line_endings.py` 在 Git fixture 中驗證修正前失敗、修正後成功，避免只在 macOS 上檢查目前工作目錄。

完整本機 135 個測試通過、原生 4 個來源雜湊通過。這不是 Windows 實機或修正後雲端 CI 通過的宣告；需將修正提交並推送，讓新 commit 的 Windows workflow 重新執行。舊 commit 的紅色檢查仍會保留。

本次證據：[回歸紀錄](evidence/windows_checkout_fix.json)、[完整測試](evidence/tests.log)。GitHub 完整 job logs 需要登入；已透過公開 job steps 確認失敗位置，並在本機重現換行問題。
