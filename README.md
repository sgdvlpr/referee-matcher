# 🧠 RefereeMatcher

**RefereeMatcher** is an AI-powered tool that recommends expert peer reviewers (referees) for a given research paper based on its abstract. It leverages topic extraction via LLMs (OpenAI or Gemini) and matches candidates using OpenAlex’s comprehensive academic graph.

---

## 🚀 Features

- 🔍 Extracts research **topics** and **keywords** from abstracts using LLMs  
- 🧠 Matches candidate referees using OpenAlex author data and publication history  
- 📊 Uses advanced scoring (recency, topic continuity, relevance) to rank authors  
- 🌐 Built with `async` support for high-performance querying  

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/sgdvlpr/referee-matcher.git
cd referee-matcher
```

---

### 2. Install dependencies

Make sure you’re using Python 3.10 or later, then install required packages:

```bash
pip install -r requirements.txt
```

---

### 3. Configure environment variables

Create your `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and set your own API keys and email:

```env
LIARA_API_KEY=your_liara_api_key
GEMINI_API_KEY=your_gemini_api_key
MAILTO=your_email@example.com
```

✅ **Do not commit your `.env` file to Git.**

---

### 4. Run the app

This is currently a script-style project. Run the matching system with:

```bash
python test_app.py
```

A FastAPI-based backend will be added soon for interactive querying.

---

## 📁 Project Structure

```
referee-matcher/
└── src/
    ├── app.py             # Core matching logic
    ├── test_app.py        # Your test or demo runner 
    ├── data/
    │   ├── openalex_fields_metadata.json
    |   ├── openalex_topics.json
    │   └── subfields.json
    ├── utils/  
    │   ├── get_all_topics.py
    ├── .env                           # Local secrets (ignored by git)
    └── .env.example                   # Sample config file for developers
├── requirements.txt
├── README.md
```

---

## ⚙️ Core Components

| Function / Class                    | Description |
|-------------------------------------|-------------|
| `query_llm_openai(prompt)`          | Queries OpenAI (GPT-4o mini) for topic/keyword extraction |
| `query_llm_gemini(prompt)`          | Uses Gemini (1.5 Flash/Pro) as an alternative |
| `get_top_referees()`                | Pulls unique top referees from a list of works |
| `build_pub_history_from_referees`   | creates a list of referees with publication history and associated recency + activity scores. |

---

## 📌 Dependencies

- `openai`
- `google-generativeai`
- `httpx`
- `python-dotenv`
- `asyncio` *(builtin)*

Install via:

```bash
pip install -r requirements.txt
```

---

## 📈 Future Plans

- 🔄 REST API via FastAPI  
- 🌐 Web frontend using React  
- 📥 Upload abstract as PDF  
- 🛡 Conflict-of-interest filtering  
- 🧠 LLM-enhanced co-reviewer suggestions  

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo  
2. Copy `.env.example` to `.env` and add your API keys  
3. Create a feature branch  
4. Submit a pull request  

Please do **not** commit `.env` or any sensitive data.

---

## 🛡 License

This project currently has **no open-source license**.  
All rights reserved © 2025 Saeed.  
Please contact the author for permission to use or contribute.

---

## 👤 Maintainer

**Saeed Gholami**  
Developer & Researcher  
📧 scramjet14@gmail.com

---

## 🌟 Acknowledgments

- [OpenAlex](https://openalex.org/)  
- [OpenAI](https://openai.com/)  
- [Gemini by Google](https://deepmind.google/technologies/gemini/)
