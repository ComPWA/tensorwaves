from . import (
    create_recipe,
    open_recipe,
)


def test_helicity():
    recipe_filename = "tests/amplitude_model_heli.yml"
    create_recipe(recipe_filename, formalism="helicity")
    recipe = open_recipe(recipe_filename)
    assert len(recipe) == 5


def test_canonical():
    recipe_filename = "tests/amplitude_model_cano.yml"
    create_recipe(recipe_filename, formalism="canonical-helicity")
    recipe = open_recipe(recipe_filename)
    assert len(recipe) == 5
