# ⚡ QUICK DEPLOYMENT GUIDE

**Time to Deploy:** 5 minutes  
**Files to Replace:** 2

---

## 🚀 STEP-BY-STEP

### 1. Backup Current Files (30 seconds)
```bash
copy core_agents.py core_agents_v15_backup.py
copy app.py app_v15_backup.py
```

### 2. Download Fixed Files (1 minute)
From this chat, download:
- ✅ `core_agents_FIXED.py`
- ✅ `app_FIXED.py`

### 3. Replace Files (30 seconds)
```bash
# Rename downloads to production names:
core_agents_FIXED.py → core_agents.py
app_FIXED.py → app.py
```

### 4. Update Version (30 seconds)
Edit `app.py`, find:
```python
Version 15 - Bug Fix: Upload Controls Display Issue
```
Change to:
```python
Version 16 - CSV Reverse Engineering Complete
```

### 5. Restart Server (30 seconds)
```bash
python app.py
```

### 6. Test (2 minutes)
1. Open http://localhost:5000
2. Domain: "Scientists"
3. Choose: "reverse"
4. Choose: "csv"
5. Upload: scientists.csv
6. Wait for completion
7. Check:
   - ✅ 3 buttons appear
   - ✅ Click "Table Viewer" → Opens
   - ✅ Click "Data Explorer" → Opens
   - ✅ Click "Ask Questions" → Opens
   - ✅ Query: "Show all scientists" → Returns data

---

## ✅ SUCCESS = All 5 checks pass!

---

## 🐛 IF SOMETHING FAILS

**Check backend logs for:**
- "Mode: Reverse Engineering" (if not, Fix #1 didn't apply)
- "Extracted entities: ['Scientist']" (if not, Fix #3/#4 issue)
- "Database exists: True" (if not, database path issue)

**Common Issues:**
1. Wrong method called → Check Fix #6 applied to app.py
2. Buttons missing → Check Fix #2 applied to core_agents.py
3. Query fails → Check Fix #5 applied to core_agents.py

**If you see errors, paste them and I'll diagnose!**

---

## 📋 WHAT GOT FIXED

**5 core_agents.py fixes:**
1. ✅ reverse=True flag
2. ✅ Artifacts in response
3. ✅ inferred_schema to state
4. ✅ database_path to state
5. ✅ DDL in response

**1 complete app.py reconstruction:**
6. ✅ Entire upload_csv function

---

**You're 5 minutes away from a working system!** 🎉

See CHANGES_SUMMARY.md for complete details.
