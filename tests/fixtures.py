"""SYNTHETIC TEST FIXTURES — invented solely to unit-test logic. NOT part of the real dataset and never shipped as data."""
from core.database import RecipeIndex
from core.preprocessing import instruction_steps, make_recipe_id

RAW = [
    ("Penne with Tomato Cream Sauce", ["1 pound penne pasta", "1 cup heavy cream", "2 cups canned tomatoes", "2 cloves garlic, minced", "1/2 cup grated parmesan cheese", "salt"], "Boil the pasta.\nSimmer tomatoes and cream.\nToss with pasta.\nTop with cheese."),
    ("Penne Arrabbiata", ["1 pound penne pasta", "2 tablespoons olive oil", "3 cloves garlic", "1 can crushed tomatoes", "red pepper flakes", "salt"], "Boil the pasta.\nCook garlic and tomatoes in oil.\nToss together."),
    ("Penne with Garlic and Oil", ["1 pound penne pasta", "1/4 cup olive oil", "4 cloves garlic", "parsley", "salt"], "Boil the pasta.\nWarm oil with garlic.\nToss with parsley."),
    ("Banana Bread", ["3 ripe bananas, mashed", "1 cup sugar", "2 large eggs", "1/3 cup butter, melted", "1 1/2 cups all-purpose flour", "1 teaspoon baking soda", "pinch of salt", "ADVERTISEMENT"], "Preheat the oven.\nMash the bananas.\nBeat in the eggs and butter.\nFold in the flour and baking soda.\nBake until golden."),
    ("Vegan Banana Bread", ["3 ripe bananas", "1 cup sugar", "1/2 cup applesauce", "1 1/2 cups all-purpose flour", "1 teaspoon baking soda", "1/3 cup vegetable oil", "salt"], "Preheat the oven.\nMash the bananas with applesauce.\nStir in the flour and oil.\nBake."),
    ("Applesauce Banana Muffins", ["2 ripe bananas", "1 cup all-purpose flour", "3/4 cup sugar", "1/2 cup applesauce", "1 teaspoon baking powder", "salt", "1/4 cup vegetable oil"], "Mix the bananas and applesauce.\nAdd flour and sugar.\nBake the muffins.\nCool."),
    ("Banana Walnut Bread", ["3 bananas", "1 cup sugar", "2 eggs", "2 cups all-purpose flour", "1 cup walnuts", "1 teaspoon baking soda", "1/2 cup butter"], "Preheat.\nMix.\nAdd eggs.\nFold in walnuts.\nBake."),
    ("Tomato Soup", ["4 tomatoes", "1 onion, chopped", "2 cloves garlic", "2 cups vegetable broth", "salt"], "Chop.\nSauté onion.\nAdd tomatoes.\nAdd broth.\nSimmer.\nBlend."),
    ("Quick Tomato Salad", ["3 tomatoes", "1 red onion", "2 tablespoons olive oil", "1 tablespoon vinegar", "salt"], "Slice.\nToss."),
    ("Tomato Rice", ["1 cup rice", "2 tomatoes", "1 onion", "2 cloves garlic", "salt", "2 tablespoons oil"], "Sauté onion.\nAdd rice.\nAdd tomatoes and water.\nCook covered."),
    ("Peanut Noodles", ["8 ounces noodles", "1/2 cup peanut butter", "2 tablespoons soy sauce", "1 clove garlic"], "Boil noodles.\nWhisk sauce.\nToss."),
    ("Garlic Onion Chicken", ["4 chicken thighs", "4 cloves garlic", "2 onions", "salt"], "Season.\nBrown chicken.\nAdd garlic and onions.\nRoast."),
    ("Fried Rice", ["2 cups cooked rice", "2 eggs", "1 onion", "2 tablespoons soy sauce", "oil"], "Heat oil.\nScramble eggs.\nAdd onion and rice.\nAdd soy sauce.\nServe."),
    ("Egg and Tomato Rice Bowl", ["2 eggs", "2 tomatoes", "1 cup rice", "1 onion"], "Cook rice.\nScramble eggs with tomatoes and onion.\nServe over rice."),
    ("French Onion Soup", ["4 onions", "3 tablespoons butter", "4 cups beef broth", "1 cup gruyere cheese"], "Slice.\nCaramelize.\nAdd broth.\nSimmer.\nTop with cheese.\nBroil."),
    ("Plain Rice", ["1 cup rice", "2 cups water", "salt"], "Rinse.\nBoil covered."),
]


def records():
    out = []
    for k, (title, ings, instr) in enumerate(RAW):
        rid = make_recipe_id("synthetic.json", f"k{k}", title, ings, instr)
        out.append({"id": rid, "title": title, "ingredients": ings, "instructions": instr, "source_key": f"k{k}"})
    return out


def make_index():
    return RecipeIndex.from_records(records())


def id_of(index, title):
    return index.ids[index.titles.index(title)]
