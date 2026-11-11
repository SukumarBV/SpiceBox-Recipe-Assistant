# 🍲 SpiceBox – AI Recipe Assistant

**Turn what you have into something delicious!**

SpiceBox is a GenAI-powered recipe assistant that helps you discover recipes, find meals using available ingredients, explore ingredient substitutions and analyse recipe data through natural-language conversations.

It combines **Gemini AI, Python and SQLite** to deliver intelligent recommendations while keeping recipe retrieval, calculations and verification grounded in the actual database.

<p align="center">

### 🚀 [Try SpiceBox Live](https://spicebox-recipe-assistant.streamlit.app/)

</p>

> **Note:** The first visit may take a little longer because the hosted application may need to wake up from a cold start.

## ✨ Features

### 🔍 Discover Recipes
- Search recipes using dish names, ingredients or natural language.
- Ask for recommendations based on food preferences and cravings.
- Refine searches through conversational follow-ups.
- Explore recipe statistics, ingredient counts and preparation steps.

### 🥗 Cook With My Ingredients
- Find recipes using ingredients you already have.
- Identify matched, missing and unused ingredients.
- Apply dietary exclusions and preparation-step limits.
- Rank recipes based on ingredient matches and preferences.

### 🔄 Ingredient Substitutions
- Explore possible ingredient alternatives.
- Use evidence from similar recipes.
- View updated ingredient lists and instructions.
- Understand potential effects and limitations of substitutions.

### 📊 Recipe Insights
- Analyse ingredient frequencies.
- Calculate recipe counts and percentages.
- Explore average and median ingredient counts.
- Rank recipes by preparation steps.

### 🤖 AI-Powered Natural Language Understanding
SpiceBox uses Gemini to interpret requests such as:

- "I'm craving something spicy."
- "Suggest a light meal for summer."
- "What can I cook with potatoes and onions?"
- "Find something vegetarian and filling."
- "Can you suggest a quicker recipe?"

The AI interprets requests, while Python and SQLite handle recipe retrieval, filtering, ranking and calculations.

**AI is optional:** SpiceBox also supports rule-based recommendations without an API key.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web application |
| SQLite | Recipe database |
| Gemini API | Natural-language understanding |
| Pandas | Not required by the application |
| Git & GitHub | Version control and hosting |

## 🧠 How It Works

```text
User's Natural Language Request
              ↓
       Gemini AI / Rules
              ↓
     Structured Search Criteria
              ↓
      Python + SQLite Search
              ↓
      Preference-Based Ranking
              ↓
       Relevant Recipes
```

The LLM interprets user intent but does not invent recipes or database results. Recipe information and calculations are retrieved and processed locally.

## 💻 Run Locally

**Requirements:** Python 3.11+

### 1. Clone the repository

```bash
git clone https://github.com/SukumarBV/SpiceBox-Recipe-Assistant.git
cd SpiceBox-Recipe-Assistant
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the recipe dataset

Download the Recipe Box datasets from:

https://eightportions.com/datasets/Recipes/

Place the following files inside `data/raw/`:

- `recipes_raw_nosource_ar.json`
- `recipes_raw_nosource_epi.json`
- `recipes_raw_nosource_fn.json`

Build the database:

```bash
python scripts/inspect_dataset.py
python scripts/build_database.py
```

### 5. Configure Gemini (Optional)

Create a `.env` file using `.env.example` and add:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key
LLM_MODEL=your_supported_gemini_model
LLM_MODE=assist
```

You can also run SpiceBox without an API key.

### 6. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🧪 Testing

Run the automated tests:

```bash
python -m pytest -q
```

The included test suite uses a synthetic recipe fixture. Real-dataset evaluation can be run separately:

```bash
python scripts/run_evaluation.py
```

## ⚠️ Limitations

- Nutrition, calories, exact cooking time and prices are not available as verified dataset fields.
- Cuisine, meal-type and food-preference classifications may rely on ingredient and title-based heuristics.
- Ingredient restrictions are text-based and are not a guarantee of allergy safety.
- Ingredient quantities and substitution outcomes are not independently verified.
- Recommendation quality depends on the underlying recipe data.

## 📂 Project Structure

```text
SpiceBox_Recipe_Assistant/
│
├── app.py
├── core/
│   ├── llm.py
│   ├── preferences.py
│   ├── search.py
│   ├── conversation.py
│   └── database.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── build_database.py
│   └── run_evaluation.py
│
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## 📚 Dataset Attribution

Recipe data is sourced from **Recipe Box by Ryan Lee**:

https://eightportions.com/datasets/Recipes/

Please refer to the dataset provider's terms for usage and redistribution conditions.

## 👩‍💻 Project

**SpiceBox – AI Recipe Assistant**

- **Live Application:** https://spicebox-recipe-assistant.streamlit.app/
- **GitHub Repository:** https://github.com/SukumarBV/SpiceBox-Recipe-Assistant

Built as part of a **GenAI Internship Take-Home Assignment**.

