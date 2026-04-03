// ══════════════════════════════════════
//  Ingredient reveal (all four)
// ══════════════════════════════════════
var ingredientAnimationState = {
  animation: null,
  ghost: null,
  locked: false,
  timerId: null
};

function getIngredientStack(n) {
  return document.querySelector('#slide-3 .ingredient-stack[data-ingredient="' + n + '"]');
}

function getIngredientPhase(n) {
  var stack = getIngredientStack(n);
  return stack ? (stack.dataset.phase || 'hidden') : 'hidden';
}

function getIngredientImage(n) {
  return document.querySelector('#ingredient' + n + 'ImageWrap .ingredient-image');
}

function getIngredientStyles(n) {
  var stack = getIngredientStack(n);
  return stack ? getComputedStyle(stack) : null;
}

function getIngredientHeroStage() {
  return document.getElementById('ingredientHeroStage');
}

function getIngredientHeroImage() {
  return document.getElementById('ingredientHeroImage');
}

function isInteractionLocked() {
  var active = document.querySelector('.slide.active');
  return !!(ingredientAnimationState.locked && active && active.id === 'slide-3');
}

function getFirstIncompleteIngredient() {
  for (var i = 1; i <= 4; i++) {
    if (getIngredientPhase(i) !== 'grid') return i;
  }
  return null;
}

function updateIngredientButtons() {
  var firstIncomplete = getFirstIncompleteIngredient();
  for (var i = 1; i <= 4; i++) {
    var btn = document.querySelector('#ingredient' + i + 'Hidden [data-ingredient]');
    if (!btn) continue;
    btn.disabled = ingredientAnimationState.locked || getIngredientPhase(i) !== 'hidden' || firstIncomplete !== i;
  }
}

function applyHeroTheme(n) {
  var stage = getIngredientHeroStage();
  var styles = getIngredientStyles(n);
  if (!stage || !styles) return;
  stage.style.setProperty('--ingredient-accent', styles.getPropertyValue('--ingredient-accent').trim());
  stage.style.setProperty('--ingredient-accent-bg', styles.getPropertyValue('--ingredient-accent-bg').trim());
  stage.style.setProperty('--ingredient-accent-border', styles.getPropertyValue('--ingredient-accent-border').trim());
}

function showIngredientHero(n) {
  var stage = getIngredientHeroStage();
  var heroImage = getIngredientHeroImage();
  var sourceImage = getIngredientImage(n);
  if (!stage || !heroImage || !sourceImage) return;

  applyHeroTheme(n);
  heroImage.src = sourceImage.getAttribute('src');
  heroImage.alt = sourceImage.getAttribute('alt');
  stage.dataset.ingredient = String(n);
  stage.classList.add('is-visible');
  stage.setAttribute('aria-hidden', 'false');
}

function hideIngredientHero() {
  var stage = getIngredientHeroStage();
  if (!stage) return;
  stage.classList.remove('is-visible');
  stage.setAttribute('aria-hidden', 'true');
  stage.removeAttribute('data-ingredient');
}

function clearIngredientFlightTimer() {
  if (ingredientAnimationState.timerId === null) return;
  clearTrackedTimeout(ingredientAnimationState.timerId);
  ingredientAnimationState.timerId = null;
}

function finalizeIngredientFlight() {
  clearIngredientFlightTimer();

  if (ingredientAnimationState.ghost && ingredientAnimationState.ghost.parentNode) {
    ingredientAnimationState.ghost.parentNode.removeChild(ingredientAnimationState.ghost);
  }
  ingredientAnimationState.ghost = null;
  ingredientAnimationState.animation = null;
  ingredientAnimationState.locked = false;
  updateIngredientButtons();
}

function cancelIngredientFlight() {
  clearIngredientFlightTimer();

  if (ingredientAnimationState.animation) {
    ingredientAnimationState.animation.onfinish = null;
    ingredientAnimationState.animation.oncancel = null;
    ingredientAnimationState.animation.cancel();
    ingredientAnimationState.animation = null;
  }

  if (ingredientAnimationState.ghost && ingredientAnimationState.ghost.parentNode) {
    ingredientAnimationState.ghost.parentNode.removeChild(ingredientAnimationState.ghost);
  }

  ingredientAnimationState.ghost = null;
  ingredientAnimationState.locked = false;
}

function setIngredientPhase(n, phase) {
  var stack = getIngredientStack(n);
  var hidden = document.getElementById('ingredient' + n + 'Hidden');
  var revealed = document.getElementById('ingredient' + n + 'Revealed');
  var imageWrap = document.getElementById('ingredient' + n + 'ImageWrap');
  var card = document.getElementById('ingredient' + n + 'Card');
  if (!stack || !hidden || !revealed || !imageWrap || !card) return;

  stack.dataset.phase = phase;
  hidden.style.display = phase === 'hidden' ? '' : 'none';
  revealed.style.display = phase === 'hidden' ? 'none' : 'flex';
  imageWrap.classList.toggle('is-target', phase === 'hero');
  imageWrap.classList.toggle('is-visible', phase === 'grid');
  imageWrap.setAttribute('aria-hidden', phase === 'grid' ? 'false' : 'true');
  card.style.borderColor = phase === 'hidden' ? '' : INGREDIENT_COLORS[n];
  updateIngredientButtons();
}

function animateIngredientToGrid(n) {
  var heroImage = getIngredientHeroImage();
  var targetImage = getIngredientImage(n);
  var styles = getIngredientStyles(n);
  if (!heroImage || !targetImage || !styles) {
    setIngredientPhase(n, 'grid');
    hideIngredientHero();
    return;
  }

  var startRect = heroImage.getBoundingClientRect();
  ingredientAnimationState.locked = true;
  updateIngredientButtons();

  setIngredientPhase(n, 'grid');
  hideIngredientHero();

  var endRect = targetImage.getBoundingClientRect();
  if (!startRect.width || !startRect.height || !endRect.width || !endRect.height) {
    finalizeIngredientFlight();
    return;
  }

  var ghost = document.createElement('div');
  ghost.className = 'ingredient-flight-ghost';
  ghost.style.left = startRect.left + 'px';
  ghost.style.top = startRect.top + 'px';
  ghost.style.width = startRect.width + 'px';
  ghost.style.height = startRect.height + 'px';
  ghost.style.setProperty('--ingredient-accent', styles.getPropertyValue('--ingredient-accent').trim());
  ghost.style.setProperty('--ingredient-accent-bg', styles.getPropertyValue('--ingredient-accent-bg').trim());
  ghost.style.setProperty('--ingredient-accent-border', styles.getPropertyValue('--ingredient-accent-border').trim());

  var ghostImage = document.createElement('img');
  ghostImage.src = heroImage.getAttribute('src');
  ghostImage.alt = heroImage.getAttribute('alt');
  ghost.appendChild(ghostImage);
  document.body.appendChild(ghost);
  ingredientAnimationState.ghost = ghost;

  var deltaX = endRect.left - startRect.left;
  var deltaY = endRect.top - startRect.top;
  var scaleX = endRect.width / startRect.width;
  var scaleY = endRect.height / startRect.height;
  var squeezeScaleX = Math.max(scaleX * 0.82, scaleX * 0.62);
  var squeezeScaleY = Math.min(scaleY * 1.12, Math.max(scaleY * 1.02, 0.48));
  var duration = 660;

  if (ghost.animate) {
    ingredientAnimationState.animation = ghost.animate([
      {
        transform: 'translate3d(0px, 0px, 0px) scale(1, 1)',
        opacity: 1,
        borderRadius: '1rem'
      },
      {
        offset: 0.64,
        transform: 'translate3d(' + (deltaX * 0.72) + 'px, ' + (deltaY * 0.72) + 'px, 0px) scale(' + squeezeScaleX + ', ' + squeezeScaleY + ')',
        opacity: 0.98,
        borderRadius: '0.82rem'
      },
      {
        transform: 'translate3d(' + deltaX + 'px, ' + deltaY + 'px, 0px) scale(' + scaleX + ', ' + scaleY + ')',
        opacity: 0.84,
        borderRadius: '0.28rem'
      }
    ], {
      duration: duration,
      easing: 'cubic-bezier(0.22, 1, 0.36, 1)',
      fill: 'forwards'
    });

    ingredientAnimationState.animation.onfinish = finalizeIngredientFlight;
    ingredientAnimationState.animation.oncancel = finalizeIngredientFlight;
    return;
  }

  ingredientAnimationState.timerId = setTrackedTimeout(finalizeIngredientFlight, duration);
}

function advanceIngredientPhase(n) {
  var phase = getIngredientPhase(n);
  if (phase === 'hidden') {
    setIngredientPhase(n, 'text');
    return true;
  }

  if (phase === 'text') {
    setIngredientPhase(n, 'hero');
    showIngredientHero(n);
    return true;
  }

  if (phase === 'hero') {
    animateIngredientToGrid(n);
    return true;
  }

  return false;
}

function revealIngredient(n) {
  if (ingredientAnimationState.locked) return;
  if (getFirstIncompleteIngredient() !== n) return;
  if (getIngredientPhase(n) !== 'hidden') return;
  setIngredientPhase(n, 'text');
}

function resetIngredients() {
  cancelIngredientFlight();
  hideIngredientHero();

  for (var i = 1; i <= 4; i++) {
    setIngredientPhase(i, 'hidden');
  }

  updateIngredientButtons();
}

function runIngredientsStep() {
  if (ingredientAnimationState.locked) return true;

  var nextIngredient = getFirstIncompleteIngredient();
  if (nextIngredient === null) return false;
  return advanceIngredientPhase(nextIngredient);
}
