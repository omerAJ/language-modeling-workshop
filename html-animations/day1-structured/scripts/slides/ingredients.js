// ══════════════════════════════════════
//  Ingredient reveal (all four)
// ══════════════════════════════════════
function revealIngredient(n) {
  document.getElementById('ingredient' + n + 'Hidden').style.display = 'none';
  document.getElementById('ingredient' + n + 'Revealed').style.display = 'block';
  document.getElementById('ingredient' + n + 'Card').style.borderColor = INGREDIENT_COLORS[n];
}

function resetIngredients() {
  for (var i = 1; i <= 4; i++) {
    document.getElementById('ingredient' + i + 'Hidden').style.display = '';
    document.getElementById('ingredient' + i + 'Revealed').style.display = 'none';
    document.getElementById('ingredient' + i + 'Card').style.borderColor = '';
  }
}

function runIngredientsStep() {
  for (var i = 1; i <= 4; i++) {
    var hidden = document.getElementById('ingredient' + i + 'Hidden');
    if (hidden && hidden.style.display !== 'none') {
      revealIngredient(i);
      return true;
    }
  }
  return false;
}
