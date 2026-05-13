# AutoResearchClaw：LLM 生成回應之 Citation 完整機制分析

> **分析對象**：`aiming-lab/AutoResearchClaw` ・ 分支 `claude/research-search-citation-8qOpv` ・ 2026-05-11
>
> **聚焦範圍**：LLM 端的 citation 機制（context 儲存、去重、引用格式、prompt、優缺點、精準引用之道）。
> 本文是 [autoresearchclaw_search_citation_analysis.html](./autoresearchclaw_search_citation_analysis.html) 的姊妹文，
> 上一篇談「文獻檢索與驗證生命週期」，這一篇談「LLM 生成端如何引用」。

---

## 目錄

1. [檢索 Context（Chunk）的儲存與管理](#1-檢索-contextchunk的儲存與管理)
2. [去重複機制](#2-去重複機制)
3. [LLM 產生回應的 Citation 呈現格式](#3-llm-產生回應的-citation-呈現格式)
4. [LLM 用以生成 Citation 的 Prompt](#4-llm-用以生成-citation-的-prompt)
5. [Citation 機制優缺點分析](#5-citation-機制優缺點分析)
6. [如何做到精準 Citation（含改進建議）](#6-如何做到精準-citation含改進建議)

---

## 1. 檢索 Context（Chunk）的儲存與管理

ARC 的 LLM 端 citation 機制建立在**三類成品（artifact）**上，並非典型的「向量資料庫 + chunk 檢索」架構。這三類成品在管線中以檔案形式持久化，並在 Stage 17 寫稿時注入 prompt。

### 1.1 三類核心 artifact

| Artifact | 產出階段 | 路徑 | 內容 | 用途 |
|---|---|---|---|---|
| `candidates.jsonl` | Stage 4 | `stage-04/candidates.jsonl` | 每列一篇候選論文：`title, authors, year, abstract, doi, arxiv_id, url, cite_key, citation_count, venue` | 提供 LLM 撰寫時的「可引用清單」 |
| `references.bib` | Stage 4 | `stage-04/references.bib` | 由 `Paper.to_bibtex()` 生成的 BibTeX | 預先驗證（P3）與最終匯出 |
| `evidence_cards.jsonl` / `stage-06/cards/*.md` | Stage 6 | `stage-06/` | 結構化證據卡：`card_id, cite_key, problem, method, data, metrics, findings, limitations` | 知識庫存檔；**目前並未直接餵入 paper_draft prompt** |

來源：`pipeline/stage_impls/_paper_writing.py:1934-1938`、`_literature.py:773-830`

### 1.2 「Chunk」實際內容

ARC **沒有將論文全文切分為向量化的 chunk**。`web/pdf_extractor.py` 雖然可以抽 PDF 文字，但結果只進入 candidates 的 `abstract` 欄位，並未做語意切片。

實際送入 LLM 撰寫 prompt 的「context」是一份扁平的 **cite_key → 摘要資訊** 對照表，於執行時動態組裝：

```python
# _paper_writing.py:1952-1964（節錄）
for row_text in candidates_text.strip().splitlines():
    row = _safe_json_loads(row_text, {})
    if isinstance(row, dict) and row.get("cite_key"):
        # 取第一作者 + et al.
        first_author = row["authors"][0]
        # ... parse author name ...
        if len(row["authors"]) > 1:
            authors_info += " et al."
        title = row.get("title", "")
        cite_lines.append(
            f"- [{row['cite_key']}] → TITLE: \"{title}\" "
            f"| {authors_info} ({row.get('venue', '')}, {row.get('year', '')}, "
            f"cited {row.get('citation_count', 0)} times) "
            f"| ONLY cite this key when discussing: {title}"
        )
```

每筆 reference 在 prompt 中呈現為一行：

```
- [vaswani2017attention] → TITLE: "Attention Is All You Need" | Vaswani et al. (NeurIPS, 2017, cited 80000 times) | ONLY cite this key when discussing: Attention Is All You Need
```

### 1.3 關鍵設計選擇

- **沒有 vector store**：cite_key 是字串型別，由 `Paper.cite_key` property 確定性生成（`models.py:57-72`）。
- **沒有 RAG retrieval at generation time**：LLM 在寫稿時看到的是「全部可引用清單」（25-60 篇），而非根據語意動態檢索的少量 chunk。
- **abstract 進 prompt 但不顯式呈現**：論文摘要存於 candidates.jsonl，作為 Stage 6 evidence cards 的輸入；paper_draft prompt 只列出 cite_key 與標題，不直接放摘要全文（避免 token 爆量）。

> **觀察**：此設計把「相關性判斷」交給 LLM 自己，僅靠 prompt 中的 `ONLY cite this key when discussing: <title>` 軟性約束。對小型文獻集合（&lt; 100 篇）可行；對大型語料則需引入向量檢索。

---

## 2. 去重複機制

ARC 的去重發生在兩個時間點，但**第二層只是隱含去重**，並未對 cite_key 命名空間做主動處理。

### 2.1 Layer 1：檢索時去重（Stage 4）

位於 `literature/search.py:270-348` 的 `_deduplicate()`：

```python
def _normalise_title(title: str) -> str:
    """小寫 + 移除標點 + 折疊空白"""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", title.lower())).strip()
```

三層比對順序：
1. **DOI**（小寫、去前綴）
2. **arXiv ID**
3. **正規化標題的字串等值比對**（**非** Jaccard 模糊比對）

若發現重複，**citation_count 較高者勝出**（偏好 Semantic Scholar 的 metadata 而非 arXiv-only）。

### 2.2 Layer 2：LLM Context 注入時的「隱含」去重

Stage 17 寫稿前，`candidates_text` 直接取用 Stage 4 已去重的結果，**沒有第二次顯式去重**（`_paper_writing.py:1934-1984` 無 `seen`/`unique` 邏輯）。

### 2.3 cite_key 碰撞風險（已知缺口）

`models.py:57-72` 的 cite_key 生成規則：

```python
@property
def cite_key(self) -> str:
    """Normalised citation key: ``lastname<year><keyword>``."""
    last = self.authors[0].last_name() if self.authors else "anon"
    yr = str(self.year) if self.year else "0000"
    kw = ""
    for word in self.title.split():
        cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
        if len(cleaned) > 3 and cleaned not in _STOPWORDS:
            kw = cleaned
            break
    return f"{last}{yr}{kw}"
```

⚠️ **沒有碰撞後綴（如 `_a`/`_b`）**。若兩篇論文的「首作者姓氏 + 年份 + 標題首個非停用詞」三者相同，cite_key 將塌縮為同一個。

> **實例**：He, Kaiming 2016 年的 *Deep Residual Learning for Image Recognition* 之 cite_key 為 `he2016deep`；
> 同年另一篇 *Deep Networks with Stochastic Depth*（Huang et al., 2016；但若我們改設想同樣由 He 主筆）也會落到 `he2016deep` ──
> 兩者首字皆為 "Deep"（4 字、非停用詞），就會碰撞。實務上凡是首作者、年份相同且標題首個非停用詞重疊的論文都會塌縮成同一鍵。

### 2.4 Layer 3：寫稿後的對賬（Stage 22）

雖非「去重」，但 `_review_publish.py:1853-1920` 確保最終 .bib 與正文引用一致：

```python
# _review_publish.py:1856-1869
_cite_key_pat = r"[a-zA-Z]+\d{4}[a-zA-Z0-9_-]*"
cited_keys_in_paper: set[str] = set()
# 單鍵 [smith2024]
for m in _re.finditer(rf"\[({_cite_key_pat})\]", final_paper):
    cited_keys_in_paper.add(m.group(1))
# 複合鍵 [smith2024, jones2023] 或 [smith2024; jones2023]
for m in _re.finditer(r"\[([^\]]{10,300})\]", final_paper):
    inner = m.group(1)
    parts = _re.split(r"[,;]\s*", inner)
    if all(_re.fullmatch(_cite_key_pat, p.strip()) for p in parts if p.strip()):
        for p in parts:
            if p.strip():
                cited_keys_in_paper.add(p.strip())
```

抽到的鍵集合會與 `valid_keys`（由 `references.bib` 用 `@\w+\{([^,]+),` 抽出）求差集；不在 .bib 的鍵會被嘗試補回或從正文移除。

---

## 3. LLM 產生回應的 Citation 呈現格式

### 3.1 LLM 端輸出：Markdown `[cite_key]` 格式

ARC 明確要求 LLM 使用方括號內含 cite_key 的格式：

```
Recent work [vaswani2017attention] has shown that self-attention layers can replace
recurrent networks, which is consistent with later findings [devlin2019bert,
liu2019roberta] on large-scale pre-training.
```

支援的變體：
| 格式 | 範例 | 處理位置 |
|---|---|---|
| 單鍵 | `[smith2024transformer]` | `_review_publish.py:1859` |
| 逗號複合鍵 | `[smith2024, jones2023]` | `_review_publish.py:1862-1869` |
| 分號複合鍵 | `[smith2024; jones2023]` | `_review_publish.py:1836-1848`（會被自動轉成逗號） |

分號→逗號的修正：

```python
# _review_publish.py:1836-1848
def _fix_semicolon_cites(m_sc):
    inner = m_sc.group(1)
    parts = [p.strip() for p in inner.split(";")]
    _ck = r"[a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9_]*"
    if all(_re.fullmatch(_ck, p) for p in parts):
        return "[" + ", ".join(parts) + "]"
    return m_sc.group(0)

final_paper = _re.sub(r"\[([^\]]+;[^\]]+)\]", _fix_semicolon_cites, final_paper)
```

### 3.2 Author-Year 漂移與回填

Stage 19 的修訂 LLM 偶爾會自作主張把 `[smith2024transformer]` 改寫成 *(Smith et al., 2024)*。`_review_publish.py:1742-1806` 的 `_build_author_year_map()` 反向恢復：

- 構建映射：`(lastname_lower, year) → cite_key`
- 用 regex 偵測自然語言引用：`(\w+) et al.?,? (\d{4})`、`(\w+) and (\w+) \((\d{4})\)` 等變體
- 命中即回填為 `[cite_key]` 格式

### 3.3 Markdown → LaTeX 轉換（Stage 22）

最終匯出時，正文裡的 `[cite_key]` 會被轉成 LaTeX `\cite{cite_key}`：

```
[vaswani2017attention]                   → \cite{vaswani2017attention}
[devlin2019bert, liu2019roberta]         → \cite{devlin2019bert,liu2019roberta}
```

`references.bib` 也會經過 BibTeX 修剪（移除未被 cite 的條目）後一併進入 `deliverables/`。

---

## 4. LLM 用以生成 Citation 的 Prompt

### 4.1 兩段式組裝

LLM 寫稿時看到的 prompt 由兩部分組合：

1. **靜態模板**：`prompts.default.yaml` 中的 `paper_draft.system` + `paper_draft.user`
2. **動態注入塊**：`_paper_writing.py:1960-1984` 在執行時即時組裝的 `citation_instruction`

### 4.2 動態注入塊（核心）

這是整個 LLM citation 機制最關鍵的 prompt 片段：

```text
AVAILABLE REFERENCES (use [cite_key] to cite in the text):
- [vaswani2017attention] → TITLE: "Attention Is All You Need" | Vaswani et al. (NeurIPS, 2017, cited 80000 times) | ONLY cite this key when discussing: Attention Is All You Need
- [devlin2019bert] → TITLE: "BERT: Pre-training of Deep Bidirectional Transformers..." | Devlin et al. (NAACL, 2019, ...) | ONLY cite this key when discussing: ...
... (25-60 行)

CRITICAL CITATION RULES:
- In the body text, cite using [cite_key] format, e.g. [smith2024transformer].
- Do NOT write a References section — it will be auto-generated from the bibliography file.
- Do NOT invent any references or arXiv IDs not in the above list.
- You may cite a subset, but NEVER fabricate citations or change arXiv IDs.
- SEMANTIC MATCHING: Before citing a reference, verify that its TITLE matches
  the concept you are discussing. Do NOT use an unrelated cite_key just
  because it sounds similar.
- If no reference in the list matches the concept you want to cite,
  write 'prior work has shown...' WITHOUT a citation, rather than using
  a mismatched reference.
- Each [cite_key] MUST correspond to the paper whose title is shown
  next to that key in the list above. Cross-check before citing.

CITATION QUANTITY & QUALITY CONSTRAINTS:
- Cite 25-40 unique references in the paper body. The Related Work
  section alone should cite at least 15 references.
- Every citation MUST be directly relevant to the paper's topic.
- DO NOT cite papers from unrelated domains (wireless communication,
  manufacturing, UAV, etc.).
- Prefer well-known, highly-cited papers over obscure ones.
- If unsure whether a paper exists or is relevant, DO NOT cite it.
```

來源：`_paper_writing.py:1960-1984`

### 4.3 靜態模板（節錄）

`prompts.default.yaml:192-243` 的 `paper_draft`（`max_tokens: 32768`）：

- **system**：頂級 ML 論文作者人格，強調 "NEVER fabricate or approximate numbers"
- **user**：注入 `{topic}, {outline}, {experiment_summary}, {citation_instruction}, ...` 等模板變數

Related Work 段落特別規範（`_paper_writing.py:415-417`）：

```
4. **Related Work** (600-800 words): organized into 3-4 thematic subsections, each
   discussing 4-5 papers with proper citations. Compare approaches, identify
   limitations, position this work.
```

### 4.4 其他相關 prompt

| Prompt | 位置 | 與 citation 的關係 |
|---|---|---|
| `knowledge_extract` | `prompts.yaml:145-156` | "IMPORTANT: If the input contains cite_key fields, preserve them exactly in the output." |
| `literature_screen` | `prompts.yaml:171-191` | 篩選時保留 cite_key、doi、arxiv_id 等欄位 |
| `paper_revision` | `prompts.yaml:262-280` | **不含 citation 規則** ⇒ 修訂時可能漂移成 author-year（依賴 Stage 22 回填） |
| `peer_review` | `prompts.yaml:281-322` | 檢查 methodology-evidence consistency，**不檢查引用真實性** |

### 4.5 Pre-Verification：寫稿前的引用淘汰

Stage 17 在組 prompt 前先呼叫 `verify_citations()`，把 hallucinated 條目從 .bib 中剔除，**避免 LLM 從已知是假的引用中取材**：

```python
# _paper_writing.py:1910-1932（節錄）
if bib_text and bib_text.strip():
    _pre_report = _verify_cit(bib_text, inter_verify_delay=0.5)
    _kept = _pre_report.verified + _pre_report.suspicious
    _removed = _pre_report.hallucinated
    if _removed > 0:
        bib_text = filter_verified_bibtex(
            bib_text, _pre_report, include_suspicious=True
        )
        (stage_dir / "references_preverified.bib").write_text(bib_text, encoding="utf-8")
        logger.info(
            "P3: Pre-verification kept %d/%d citations (removed %d hallucinated)",
            _kept, _pre_report.total, _removed,
        )
```

---

## 5. Citation 機制優缺點分析

### 5.1 優點

| 面向 | 設計 | 效益 |
|---|---|---|
| **多層防幻覺** | 預驗證（P3）+ 寫稿規則 + 最終驗證（Stage 23）+ Stage 22 對賬 | 任一層偵測到 hallucination 即可攔截，整體覆蓋率高 |
| **零向量資料庫依賴** | cite_key + 標題清單直接放進 prompt | 部署簡單；無需 embedding model 與向量索引 |
| **明確的引用規則** | `SEMANTIC MATCHING`、`ONLY cite this key when discussing: <title>` 寫死於 prompt | LLM 較不易張冠李戴 |
| **支援複合引用** | `[a, b, c]`、`[a; b]` 兩種格式都會被處理 | 多源歸因有語法支援 |
| **author-year 漂移自動修復** | `_build_author_year_map()` 反向回填 | 修訂階段風格漂移不會留下「找不到 cite_key」的破窗 |
| **可引用清單範圍受控** | 只能從 25-60 篇預先驗證的論文中挑 | 不會生成完全捏造的 cite_key（被 Stage 22 偵測即移除） |

### 5.2 缺點與盲點

| 面向 | 問題 | 影響 |
|---|---|---|
| **cite_key 碰撞無處理** | `models.py:57-72` 不加後綴 | 同作者同年同首關鍵字 → 兩篇論文共用一個 cite_key，正文無法區辨 |
| **無 chunk-level retrieval** | 全部 reference 一次放進 prompt | 對大型語料（&gt; 100 篇）會 token 爆量；且 LLM 對「列表後段」記憶較弱 |
| **abstract 未進寫稿 prompt** | 只列 title | LLM 必須靠標題猜內容；標題曖昧時容易 mismatch |
| **無 under-citation 偵測** | 沒有檢查「這段論述明顯需要 citation 卻沒給」的機制 | LLM 可能整段論述未引用任何來源（潛在學術不端風險） |
| **peer_review 不檢查引用真實性** | 同儕審查 prompt 只查 methodology-evidence consistency | 漏掉 reference 層次的審查 |
| **revision prompt 缺 citation 規則** | Stage 19 修訂時引用可能漂移 | 雖然 Stage 22 會回填，但對未在映射表中的情況失效 |
| **多源歸因無 quota** | 沒有「每個論述至少需 N 篇 supporting cite」的硬性檢查 | LLM 可能僅引一篇，遺漏其他相關來源 |
| **SUSPICIOUS 預設保留** | `filter_verified_bibtex(include_suspicious=True)` | 半幻覺條目可能進入論文 |
| **規則式輸出依賴 LLM 配合** | "DO NOT invent" 是軟性指令 | 若 LLM 強行造鍵，僅靠後處理移除（會留下「裸論述」） |

### 5.3 安全網覆蓋率評估

ARC 的引用安全網可分為「事前」與「事後」兩類：

```
事前（寫稿前）：
  ✅ 文獻來源限定為已驗證的 .bib（P3 預驗證）
  ✅ Prompt 明確列出唯一可用的 cite_key
  ❌ 沒有 retrieval-based context narrowing

事後（寫稿後）：
  ✅ 從正文抽 cite_key 與 .bib 差集
  ✅ 找不到的 cite_key 嘗試 API 補回
  ✅ Stage 23 最終四層驗證
  ❌ 沒有 "claim → required citations" 反向稽核
  ❌ 沒有 "this paragraph has 0 citations but should have X" 偵測
```

**結論**：ARC 在「不引用假來源」這件事上做得很好；但在「該引用的有沒有都引用到」上有明顯缺口。

---

## 6. 如何做到精準 Citation（含改進建議）

「精準引用」可拆成兩個正交目標：

- **No fabrication**（不引假）：引用的論文真實存在且與論述相關 ── ARC 目前已做得相當好
- **No omission**（不漏引）：當論述參考多份來源時，全部來源都要被引用 ── ARC 目前**有明顯缺口**

以下提出可在現有架構上漸進落地的 6 項改進。

### 6.1 為每個論述（claim）做「需 citation 的反向稽核」

**現況缺口**：peer_review 不檢查 under-citation；Stage 22 只做正向（cite_key→.bib）核對。

**建議**：在 Stage 18 peer_review 加入一個新角色 *Citation Auditor*，職責：

1. 用句子斷句器（或 LLM）抽出每個 *factual claim*（含因果、數據、方法宣稱）。
2. 對每個 claim 跑 retrieval 比對 `evidence_cards.jsonl`，找出潛在 supporting cite_key。
3. 若 claim 出處 ≥ 2 但正文僅引 0–1 個，列入 `under_cited.json`，由 Stage 19 補引。

可重用既有的 `hitl/claim_verifier.py:84-109`（已能抽 citation/numerical/comparative/existence claim）作為斷句基礎。

### 6.2 從「列表 prompt」升級到「按段落 retrieval」

**現況缺口**：寫稿時全部 25-60 篇 reference 一次塞進 prompt，LLM 對列表後半段記憶較弱。

**建議**：為每個段落（Introduction、Related Work、Method、Results、Discussion）分別計算 top-K relevant cite_keys：

```python
# 假想 API
def select_refs_for_section(section_name: str, draft_outline: str,
                            candidates: list[Paper], top_k: int = 12) -> list[Paper]:
    # 1. 用 embedding 算 (section_outline) vs (paper.title + paper.abstract) 餘弦相似度
    # 2. 或退化為關鍵字 TF-IDF
    return sorted(candidates, key=score, reverse=True)[:top_k]
```

只把與該段落相關的 top-K 餵入該段落的 generation prompt。這也是業界 RAG 的標準做法。

### 6.3 cite_key 碰撞防護

**現況缺口**：`models.py:57-72` 不加後綴。

**建議**：在 Stage 4 收論文後，對 `[paper.cite_key for paper in collected]` 做計數；發現碰撞時，第二個之後依序加 `b`/`c` 後綴：

```python
def deduplicate_cite_keys(papers: list[Paper]) -> list[Paper]:
    seen: dict[str, int] = {}
    result = []
    for p in papers:
        base = p.cite_key
        if base not in seen:
            seen[base] = 1
            result.append(p)
        else:
            seen[base] += 1
            suffix = chr(ord('a') + seen[base] - 1)  # b, c, d, ...
            new_key = f"{base}{suffix}"
            result.append(replace(p, _cite_key_override=new_key))
    return result
```

需同步擴充 `Paper` dataclass 增加 `_cite_key_override` 欄位。

### 6.4 在 revision prompt 明確保留引用格式

**現況缺口**：`paper_revision`（prompts.yaml:262-280）沒有 citation 規則 ⇒ LLM 自由發揮 ⇒ 變成 (Smith et al., 2024)，事後再回填。

**建議**：在 revision prompt 直接加入：

```
CRITICAL: Preserve all existing [cite_key] markers verbatim. Do not convert
them to author-year format. If you reorganize sentences, move the marker
along with its claim.
```

可省去 `_build_author_year_map()` 的部分工作量，並減少回填失敗的邊界情況。

### 6.5 把 evidence_cards 餵入寫稿 prompt（小範圍）

**現況缺口**：evidence_cards 只存於檔案，未進入寫稿 context；LLM 只看到標題，難以判斷一篇論文具體說了什麼。

**建議**：在 `citation_instruction` 裡，為每個 cite_key 附上 evidence card 的 `findings` 一句話摘要（至多 30 字）：

```
- [vaswani2017attention] → TITLE: "Attention Is All You Need"
  KEY FINDING: Self-attention alone (without recurrence) achieves SOTA on translation.
  ONLY cite this key when discussing: self-attention, transformer architecture.
```

如此 LLM 的 *SEMANTIC MATCHING* 規則才有真正的語意基礎，不只靠標題猜。

### 6.6 多源歸因的最低引用數量規則

**現況缺口**：可選擇引一篇或多篇，沒有對「論述要多源支持」的硬性檢查。

**建議**：在 peer_review prompt 加入一條：

```
For each factual claim in the paper, check whether it draws from multiple sources.
If a claim is supported by multiple papers in the references, the paper should
cite ALL of them. Flag claims that cite only 1 source when ≥2 are available.
Return JSON: {"under_cited_claims": [{"claim": "...", "current_cites": [...],
"missing_cites": [...]}]}.
```

並在 Stage 19 revision 階段消費此 JSON 補引。

### 6.7 綜合機制總結（建議落地優先序）

| 優先序 | 改進 | 預期效益 | 實作成本 |
|---|---|---|---|
| ★★★ | 6.3 cite_key 碰撞防護 | 修正潛在的「引用塌縮」bug | 低（&lt; 30 行） |
| ★★★ | 6.4 revision prompt 強化 | 減少 author-year 漂移 | 極低（改 prompt） |
| ★★ | 6.5 evidence_card 摘要進 prompt | 提升 SEMANTIC MATCHING 準確性 | 低（30-50 行） |
| ★★ | 6.1 反向稽核 (Citation Auditor) | 偵測 under-citation | 中（新增一個 review role） |
| ★ | 6.2 段落級 retrieval | 大規模語料下的可擴展性 | 高（引入 embedding） |
| ★ | 6.6 多源歸因 quota | 避免遺漏引用 | 中（peer_review prompt + revision loop） |

---

## 附錄：關鍵原始碼定位速查

| 主題 | 檔案 | 行數 |
|---|---|---|
| Paper 資料模型 & cite_key | `researchclaw/literature/models.py` | 32-152 |
| 檢索去重 | `researchclaw/literature/search.py` | 270-348 |
| 寫稿前預驗證（P3） | `researchclaw/pipeline/stage_impls/_paper_writing.py` | 1910-1932 |
| 動態 citation_instruction 注入 | `researchclaw/pipeline/stage_impls/_paper_writing.py` | 1952-1984 |
| paper_draft 靜態 prompt | `prompts.default.yaml` | 192-243 |
| 寫稿後對賬 | `researchclaw/pipeline/stage_impls/_review_publish.py` | 1853-1920 |
| Author-year 回填 | `researchclaw/pipeline/stage_impls/_review_publish.py` | 1742-1806 |
| 分號→逗號修正 | `researchclaw/pipeline/stage_impls/_review_publish.py` | 1836-1848 |
| Claim verifier | `researchclaw/hitl/claim_verifier.py` | 84-165 |
| Citation 四層驗證 | `researchclaw/literature/verify.py` | 663-859 |

---

*本文檔由閱讀原始碼於 2026-05-11 產生。如 ARC 升級至 v0.5+，建議重新覆核行號。*
