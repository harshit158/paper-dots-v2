# PaperDots — MVP Plan

**Goal:** the smallest useful version of PaperDots — a collaborative workspace for reading papers and building a personal/shared knowledge graph from them.

**Core loop:** Add paper → Read → Highlight/Create insight → Connect insight to another concept → View knowledge map → Share. Make this loop enjoyable before adding AI.

## Scope

**Must have**

* **User** — simple auth + profile
* **Library** — add a paper (title, authors, abstract, URL, PDF), list saved papers
* **Reader** — display PDF, page navigation, select text → create insight
* **Insights** — text, source paper, page number, optional comment, creator; edit/delete own
* **Connections** — link insight/concept to another (e.g. RAG → Dense Retrieval → ColBERT)
* **Knowledge map** — interactive graph, nodes = insights, edges = connections, click node → insight
* **Sharing** — paper can be public; others view insights and connections; no complex permissions yet

**Not in MVP:** AI summarization or question generation, recommendations, automatic graph extraction, semantic search, vector DB/embeddings, multi-agent workflows, complex permissions, teams, notifications, mobile app, payments, browser extension, social feed, advanced analytics.

---

## Product roadmap

Build complexity progressively. Each stage should be useful on its own before moving to the next.

| Stage | Product | Complexity | What you learn |
| --- | --- | --- | --- |
| 0 | Personal paper notebook | 🟢 | Product + UX |
| 1 | Paper library | 🟢 | Streamlit + SQLite |
| 2 | Annotated reader | 🟢🟡 | PDF handling + state management |
| 3 | Connected notes | 🟡 | Graph modeling + Streamlit visualization |
| 4 | Shareable knowledge | 🟡 | Collaboration + permissions |
| 5 | AI reading assistant | 🟡🟠 | LLM + RAG |
| 6 | Personal Knowledge Model | 🟠 | Knowledge graphs + retrieval |
| 7 | Active learning engine | 🟠🔴 | Question generation + memory |
| 8 | Recommendation engine | 🔴 | Personalization |
| 9 | Collaborative intelligence | 🔴 | Multi-user knowledge |
| 10 | Full PaperDots platform | 🔴🔴 | Complex product/system |

The current MVP covers stages 0–4. Stages 5–10 come only after the core reading, insight, connection, and sharing loop is working.

---

## Stack

Use the simplest possible POC stack: **Python + Streamlit + SQLite**. Use Streamlit session state for the current UI state and local file storage for PDFs. Add small Python libraries only when needed for PDF rendering and graph visualization.

There is no separate frontend, backend, REST API, or React application. Keep everything in one Streamlit app and one repository. Deployment can wait until the workflow is validated.

## Data model

* **User:** id, name, email, created_at
* **Paper:** id, title, authors, abstract, url, pdf_path, owner_id, public, created_at
* **Insight:** id, paper_id, user_id, text, comment, page_number, created_at, updated_at
* **Connection:** id, source_insight_id, target_insight_id, label, created_at

No graph database — the relational tables are sufficient; the graph is a view over them.

---

## Streamlit app views

Use a simple sidebar to switch between a few views rather than building routes or separate frontend pages:

* **Home** — "Read papers. Capture ideas. Connect knowledge." + [Start Reading]
* **Library** — add a paper, upload a PDF, and view saved papers
* **Reader** — display the selected PDF, navigate pages, create and manage insights
* **Knowledge map** — view and connect insights in an interactive graph
* **Share** — generate a simple read-only view or export for a paper

Button clicks and forms call Python functions directly. SQLite handles persistence; `st.session_state` handles temporary UI state.

---

## Implementation order

Don't build everything at once — each phase is a working vertical slice.

| Phase | Work | Checkpoint |
| --- | --- | --- |
| 1 — Skeleton | Streamlit app, SQLite database, sidebar navigation, basic layout | `streamlit run` opens the app |
| 2 — Library | Paper model, add-paper form, PDF upload/storage, paper list and selection | I can add a paper and see it in my library |
| 3 — Reader | Render PDF, page navigation, text selection or manual excerpt entry, [Add Insight] | I can open a paper and capture something interesting |
| 4 — Insights | Insight model, save selected text + page number + optional comment, display beside paper, edit/delete | I can read a paper and build a collection of my own insights (first genuinely useful version) |
| 5 — Connections | Connection model, Connect action, pick source + target, optional label, display | my notes are no longer isolated |
| 6 — Knowledge map | insights → nodes, connections → edges, Streamlit graph visualization, click/select node | I can see how ideas from papers connect |
| 7 — Sharing | `Paper.public` boolean, read-only Streamlit view or export (view paper + insights + map, no editing) | I can share my understanding of a paper |

Don't optimize PDF processing or implement graph algorithms yet.

---

## Definition of done

1. User opens the Streamlit app (add authentication only if needed)
2. User adds a paper
3. User opens it and selects an interesting sentence
4. User creates several insights from selections
5. User connects two insights
6. User opens the knowledge map and sees the connected ideas
7. User shares the paper; another person can view the paper, insights, and map

If this works smoothly, **stop building**. Don't immediately add AI.

## First AI feature (only after the manual workflow works)

One capability: **"Help me understand this paper."** Given a selected passage, generate an explanation, why it matters, a related concept, and a possible connection. The user decides whether to save the result.

AI is an assistant to the reader, not the creator of the knowledge graph.

## Engineering principles

1. **Optimize for shipping:** Streamlit > React + FastAPI, SQLite > Postgres, direct Python functions > API endpoints, manual connections > automatic extraction, local PDF handling.
2. **No premature abstraction:** one feature end-to-end before building reusable frameworks.
3. **Keep the database boring:** the knowledge graph is a view over relational data.
4. **Keep AI out of the critical path:** the product must work without an LLM.
5. **Build vertical slices:** not "all infrastructure first", but "add paper → working Streamlit view", then the next feature.

---

## First 3 days

* **Day 1:** Streamlit app, SQLite database, Paper model, add-paper form, Library view → *add a paper and see it in the UI*.
* **Day 2:** PDF upload + storage + viewer, Reader view, text selection or excerpt entry, Insight model → *select text and save an insight*.
* **Day 3:** display/edit/delete insights, Connection model, connect insights, graph visualization → *read → capture → connect → visualize*.

That is the first PaperDots.

## Product principle

Not "an AI tool that summarizes research papers", but **"Kindle for building your understanding of research."**

The differentiated asset is not the PDF viewer or the LLM — it is the growing network of human-curated insights and connections across papers.
