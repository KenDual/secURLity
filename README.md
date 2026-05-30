```
╔──────────────────────────────────────────────────────────────────────╗
│███████╗███████╗=██████╗██╗===██╗██████╗=██╗=====██╗████████╗██╗===██╗│
│██╔════╝██╔════╝██╔════╝██║===██║██╔══██╗██║=====██║╚══██╔══╝╚██╗=██╔╝│
│███████╗█████╗==██║=====██║===██║██████╔╝██║=====██║===██║====╚████╔╝=│
│╚════██║██╔══╝==██║=====██║===██║██╔══██╗██║=====██║===██║=====╚██╔╝==│
│███████║███████╗╚██████╗╚██████╔╝██║==██║███████╗██║===██║======██║===│
│╚══════╝╚══════╝=╚═════╝=╚═════╝=╚═╝==╚═╝╚══════╝╚═╝===╚═╝======╚═╝===│
╚──────────────────────────────────────────────────────────────────────╝
        __  ,          ___            _
       ( /,/          ( / \          //
        /<   _  _ _    /  /, , __,  // 
   by  /  \_(/_/ / /_(/\_/(_/_(_/(_(/_  — the one from the hood
```
<p>
  <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExenF6bms2enlvYWk1M2V3c3JhanNmYnFnNWI1Ym83b3VwZzhrZmpvNiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/IeKgCDlpTqRQbZEhBF/giphy.gif" alt="secURLity demo" width="400"/>
</p>

**secURLity** detects malicious URLs before they do damage — phishing links, malware droppers, botnet C2s, and more.

Built on **19.6 million real-world URLs** from live threat intelligence feeds and the open web, secURLity doesn't rely on static blocklists or hand-written rules. It uses three independent ML models that each analyze a URL from a different angle, then combines their signals into a single verdict — with a clear explanation of *why*.

---

## What it does

- Classifies any URL as **benign or malicious** in under 2ms
- Explains every decision — not just a score, but *which parts* of the URL triggered the alert
- Three models, three perspectives — sequence patterns, technical features, and raw page content
- Trained on real data from 13+ global threat intel feeds — no synthetic noise

---

## Models

**CNN-LSTM** — reads the URL character by character, learning sequence patterns and structural signals that distinguish malicious URLs from legitimate ones.

**XGBoost** — analyzes 105 handcrafted technical features extracted from the URL: entropy, TLD distribution, digit density, charset anomalies, and more. Fast, accurate, and fully explainable via SHAP values.

**SGDClassifier** *(extra option)* — fetches the raw HTML of the page and scans its content directly. Catches phishing kits that hide behind innocent-looking URLs. Activate when you need that extra layer of certainty.

---

## Who it's for

Security analysts, threat researchers, and developers who need URL scanning that goes beyond "is this on a blocklist?" — and need to understand the reasoning behind every flag.

Every prediction comes with an explanation. No black boxes. No blind trust.

---

## Contact

Questions, feedback, or collaboration:

**maiphuhai123@gmail.com**

---

> *Because a URL is often the first thing that goes wrong.* 😬
