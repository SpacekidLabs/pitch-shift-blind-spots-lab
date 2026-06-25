from __future__ import annotations

from html import escape
import json
from pathlib import Path

import pandas as pd


RATING_FIELDS = [
    ("artifact_rating_1_bad_5_clean", "Artifacts", "1 = messy, 5 = clean"),
    ("transient_rating_1_smeared_5_crisp", "Transients", "1 = smeared, 5 = crisp"),
    ("groove_rating_1_bad_5_good", "Groove", "1 = off, 5 = solid"),
    ("overall_rating_1_bad_5_good", "Overall", "1 = poor, 5 = useful"),
]


def _rating_buttons(field: str, blind_id: str) -> str:
    buttons = []
    for value in range(1, 6):
        element_id = f"{blind_id}_{field}_{value}"
        buttons.append(
            f"""
            <label class="score-pill" for="{escape(element_id)}">
              <input id="{escape(element_id)}" name="{escape(blind_id)}_{escape(field)}" type="radio"
                     data-field="{escape(field)}" value="{value}">
              <span>{value}</span>
            </label>
            """
        )
    return "\n".join(buttons)


def _stimulus_card(row: pd.Series) -> str:
    blind_id = str(row["blind_id"])
    wav_file = str(row["wav_file"])
    shift = int(row["shift_semitones"])
    ratings = []
    for field, label, helper in RATING_FIELDS:
        ratings.append(
            f"""
            <div class="rating-row">
              <div>
                <strong>{escape(label)}</strong>
                <small>{escape(helper)}</small>
              </div>
              <div class="score-group">
                {_rating_buttons(field, blind_id)}
              </div>
            </div>
            """
        )
    return f"""
    <article class="stimulus-card" data-blind-id="{escape(blind_id)}" data-wav-file="{escape(wav_file)}"
             data-shift="{shift}">
      <header>
        <div>
          <p class="eyebrow">Blind file</p>
          <h2>{escape(blind_id)}</h2>
        </div>
        <span class="shift-badge">{shift:+d} semitones</span>
      </header>
      <audio controls preload="metadata" src="{escape(wav_file)}"></audio>
      <div class="ratings">
        {"".join(ratings)}
      </div>
      <label class="notes-label" for="{escape(blind_id)}_notes">Notes</label>
      <textarea id="{escape(blind_id)}_notes" data-field="notes"
                placeholder="What do you hear? Cymbal smear, tom wobble, groove drag, metallic ring..."></textarea>
    </article>
    """


def save_blind_listening_session(
    manifest: pd.DataFrame,
    output_path: str | Path,
    title: str = "Experiment 22 Blind Drum Listening Session",
    storage_key: str = "psbsl_exp22_scores_v1",
    export_filename: str = "exp22_drum_listening_scores.csv",
) -> None:
    output_path = Path(output_path)
    cards = "\n".join(_stimulus_card(row) for _, row in manifest.iterrows())
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --paper: #f7f2e8;
      --ink: #1f2525;
      --muted: #66706d;
      --line: #ded4c3;
      --card: #fffaf0;
      --accent: #265d63;
      --accent-2: #c86f43;
      --accent-3: #132f32;
      --shadow: 0 18px 50px rgba(38, 31, 21, 0.13);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(200, 111, 67, 0.18), transparent 34rem),
        linear-gradient(135deg, #f8f1e5 0%, #eee4d4 52%, #e7dac7 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.45;
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 42px 0 64px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 24px;
      align-items: stretch;
      margin-bottom: 24px;
    }}
    .panel, .stimulus-card {{
      background: rgba(255, 250, 240, 0.88);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}
    .intro {{
      padding: 30px;
      position: relative;
      overflow: hidden;
    }}
    .intro:after {{
      content: "";
      position: absolute;
      width: 220px;
      height: 220px;
      border: 1px solid rgba(38, 93, 99, 0.2);
      border-radius: 999px;
      right: -80px;
      top: -70px;
    }}
    .eyebrow {{
      margin: 0 0 8px;
      color: var(--accent-2);
      font: 700 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      max-width: 720px;
      font-size: clamp(34px, 5vw, 64px);
      line-height: 0.96;
      letter-spacing: -0.045em;
    }}
    .intro p {{
      max-width: 740px;
      color: var(--muted);
      font-size: 17px;
    }}
    .reference {{
      padding: 24px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 18px;
    }}
    audio {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      background: var(--accent);
      color: #fffaf0;
      cursor: pointer;
      font: 700 13px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0.02em;
    }}
    button.secondary {{
      background: transparent;
      color: var(--accent-3);
      border: 1px solid var(--line);
    }}
    .progress {{
      margin: 20px 0 26px;
      padding: 16px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    .progress-bar {{
      flex: 1;
      height: 12px;
      border-radius: 999px;
      background: #e5d8c5;
      overflow: hidden;
    }}
    .progress-fill {{
      height: 100%;
      width: 0;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      transition: width 180ms ease;
    }}
    .stimulus-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .stimulus-card {{
      padding: 22px;
    }}
    .stimulus-card header {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .stimulus-card h2 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: -0.03em;
    }}
    .shift-badge {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 10px;
      color: var(--accent);
      font: 700 12px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: nowrap;
    }}
    .ratings {{
      display: grid;
      gap: 12px;
      margin: 18px 0;
    }}
    .rating-row {{
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 12px;
      align-items: center;
      padding-top: 12px;
      border-top: 1px solid rgba(222, 212, 195, 0.82);
    }}
    .rating-row small {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .score-group {{
      display: flex;
      gap: 6px;
      justify-content: flex-end;
    }}
    .score-pill input {{
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }}
    .score-pill span {{
      display: grid;
      place-items: center;
      width: 36px;
      height: 34px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #fbf5ea;
      cursor: pointer;
      font: 700 13px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .score-pill input:checked + span {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    .notes-label {{
      display: block;
      color: var(--muted);
      margin-bottom: 6px;
      font: 700 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    textarea {{
      width: 100%;
      min-height: 82px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fffdf7;
      padding: 12px;
      color: var(--ink);
      font: 14px/1.4 ui-sans-serif, system-ui, sans-serif;
    }}
    .footer-note {{
      margin-top: 26px;
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 860px) {{
      .hero, .stimulus-grid {{
        grid-template-columns: 1fr;
      }}
      .rating-row {{
        grid-template-columns: 1fr;
      }}
      .score-group {{
        justify-content: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="panel intro">
        <p class="eyebrow">Pitch Shift Blind Spots Lab</p>
        <h1>{escape(title)}</h1>
        <p>
          Score what you hear before opening the private answer key. The page saves your ratings in this browser
          and exports a CSV when you are done.
        </p>
        <div class="actions">
          <button id="exportCsv">Export scores CSV</button>
          <button id="copyCsv" class="secondary">Copy CSV</button>
          <button id="clearScores" class="secondary">Clear local scores</button>
        </div>
      </div>
      <aside class="panel reference">
        <div>
          <p class="eyebrow">Reference</p>
          <h2>Original excerpt</h2>
          <p>Listen here first, then score the blind files below.</p>
        </div>
        <audio controls preload="metadata" src="audio/REFERENCE_original_excerpt.wav"></audio>
      </aside>
    </section>

    <section class="panel progress" aria-live="polite">
      <strong id="progressText">0 of {len(manifest)} completed</strong>
      <div class="progress-bar"><div id="progressFill" class="progress-fill"></div></div>
    </section>

    <section class="stimulus-grid">
      {cards}
    </section>

    <p class="footer-note">
      Keep `private_answer_key.csv` closed until after exporting your scores. Metrics are diagnostics, not the judge.
    </p>
  </main>

  <script>
    const STORAGE_KEY = {json.dumps(storage_key)};
    const ratingFields = {json.dumps(RATING_FIELDS)}.map(item => item[0]);
    const cards = Array.from(document.querySelectorAll(".stimulus-card"));

    function emptyRecord(card) {{
      return {{
        blind_id: card.dataset.blindId,
        wav_file: card.dataset.wavFile,
        shift_semitones: card.dataset.shift,
        artifact_rating_1_bad_5_clean: "",
        transient_rating_1_smeared_5_crisp: "",
        groove_rating_1_bad_5_good: "",
        overall_rating_1_bad_5_good: "",
        notes: ""
      }};
    }}

    function loadScores() {{
      try {{
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");
      }} catch (error) {{
        return {{}};
      }}
    }}

    function saveScores(scores) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(scores));
    }}

    function collectCard(card) {{
      const record = emptyRecord(card);
      ratingFields.forEach(field => {{
        const selected = card.querySelector(`input[data-field="${{field}}"]:checked`);
        record[field] = selected ? selected.value : "";
      }});
      const notes = card.querySelector("textarea[data-field='notes']");
      record.notes = notes ? notes.value : "";
      return record;
    }}

    function collectAll() {{
      const scores = {{}};
      cards.forEach(card => {{
        scores[card.dataset.blindId] = collectCard(card);
      }});
      return scores;
    }}

    function restoreScores() {{
      const scores = loadScores();
      cards.forEach(card => {{
        const record = scores[card.dataset.blindId];
        if (!record) return;
        ratingFields.forEach(field => {{
          if (!record[field]) return;
          const input = card.querySelector(`input[data-field="${{field}}"][value="${{record[field]}}"]`);
          if (input) input.checked = true;
        }});
        const notes = card.querySelector("textarea[data-field='notes']");
        if (notes) notes.value = record.notes || "";
      }});
      updateProgress();
    }}

    function updateProgress() {{
      const scores = collectAll();
      let complete = 0;
      Object.values(scores).forEach(record => {{
        const hasAllRatings = ratingFields.every(field => record[field]);
        if (hasAllRatings) complete += 1;
      }});
      document.getElementById("progressText").textContent = `${{complete}} of ${{cards.length}} completed`;
      document.getElementById("progressFill").style.width = `${{(complete / cards.length) * 100}}%`;
    }}

    function csvEscape(value) {{
      const text = String(value ?? "");
      if (/[",\\n]/.test(text)) return `"${{text.replaceAll('"', '""')}}"`;
      return text;
    }}

    function toCsv() {{
      const records = cards.map(card => collectCard(card));
      const headers = Object.keys(records[0]);
      return [headers.join(","), ...records.map(record => headers.map(header => csvEscape(record[header])).join(","))].join("\\n");
    }}

    function exportCsv() {{
      const blob = new Blob([toCsv()], {{ type: "text/csv;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = {json.dumps(export_filename)};
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }}

    cards.forEach(card => {{
      card.addEventListener("change", () => {{
        saveScores(collectAll());
        updateProgress();
      }});
      card.addEventListener("input", () => {{
        saveScores(collectAll());
        updateProgress();
      }});
    }});

    document.getElementById("exportCsv").addEventListener("click", exportCsv);
    document.getElementById("copyCsv").addEventListener("click", async () => {{
      try {{
        await navigator.clipboard.writeText(toCsv());
        alert("Scores CSV copied to clipboard.");
      }} catch (error) {{
        alert("Clipboard access was blocked. Use Export scores CSV instead.");
      }}
    }});
    document.getElementById("clearScores").addEventListener("click", () => {{
      if (!confirm("Clear saved scores for this browser?")) return;
      localStorage.removeItem(STORAGE_KEY);
      cards.forEach(card => {{
        card.querySelectorAll("input[type='radio']").forEach(input => input.checked = false);
        const notes = card.querySelector("textarea");
        if (notes) notes.value = "";
      }});
      updateProgress();
    }});

    restoreScores();
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
