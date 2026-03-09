function initAttentionWeightsSlide() {
  const slide = document.getElementById('slide-21');
  const tokenRow = document.getElementById('attn21-token-row');
  const rawRow = document.getElementById('attn21-raw-row');
  const scaledRow = document.getElementById('attn21-scaled-row');
  const weightRow = document.getElementById('attn21-weight-row');
  const sumChip = document.getElementById('attn21-sum-chip');
  const queryToken = document.getElementById('attn21-query-token');
  if (!slide || !tokenRow || !rawRow || !scaledRow || !weightRow || !sumChip) return;

  if (!state.attentionWeights.initialized) {
    tokenRow.innerHTML = '';
    rawRow.innerHTML = '';
    scaledRow.innerHTML = '';
    weightRow.innerHTML = '';
    if (queryToken) queryToken.textContent = ATTN_WGT_FOCUS;

    const maxWeight = Math.max.apply(null, ATTN_WGT_WEIGHTS);
    const safeMaxWeight = maxWeight > 0 ? maxWeight : 1;

    ATTN_WGT_TOKENS.forEach((token, idx) => {
      const tokenCell = createEl('div', {
        className: 'attn21-token-cell' + (token === ATTN_WGT_FOCUS ? ' is-focus' : ''),
        dataset: { token }
      });
      const chipWrap = createEl('div', { className: 'attn19-chip-wrap attn21-chip-wrap' });
      chipWrap.appendChild(createEl('div', {
        className: 'attn19-token-chip',
        id: 'attn21-chip-' + token,
        text: token
      }));
      tokenCell.appendChild(chipWrap);
      tokenRow.appendChild(tokenCell);

      const rawCell = createEl('div', { className: 'attn21-score-cell', dataset: { token } });
      rawCell.appendChild(createEl('div', {
        className: 'attn21-num-chip',
        id: 'attn21-raw-' + token,
        text: ATTN_WGT_RAW_SCORES[idx].toFixed(1)
      }));
      rawCell.appendChild(createEl('div', {
        className: 'attn21-token-mini',
        text: token
      }));
      rawRow.appendChild(rawCell);

      const scaledCell = createEl('div', { className: 'attn21-score-cell', dataset: { token } });
      scaledCell.appendChild(createEl('div', {
        className: 'attn21-num-chip',
        id: 'attn21-scaled-' + token,
        text: ATTN_WGT_SCALED_SCORES[idx].toFixed(2)
      }));
      scaledCell.appendChild(createEl('div', {
        className: 'attn21-token-mini',
        text: token
      }));
      scaledRow.appendChild(scaledCell);

      const weight = ATTN_WGT_WEIGHTS[idx];
      const fillPct = (weight / safeMaxWeight) * 100;
      const wCell = createEl('div', {
        className: 'attn21-weight-cell' + (weight === maxWeight ? ' is-peak' : ''),
        dataset: { token }
      });
      wCell.appendChild(createEl('div', {
        className: 'attn21-weight-value',
        text: weight.toFixed(2)
      }));
      const wBar = createEl('div', { className: 'attn21-weight-bar' });
      const wFill = createEl('span', { className: 'attn21-weight-fill' });
      wFill.style.setProperty('--attn21-fill', fillPct.toFixed(1) + '%');
      wBar.appendChild(wFill);
      wCell.appendChild(wBar);
      wCell.appendChild(createEl('div', {
        className: 'attn21-weight-token',
        text: token
      }));
      weightRow.appendChild(wCell);
    });

    const sum = ATTN_WGT_WEIGHTS.reduce((acc, value) => acc + value, 0);
    sumChip.innerHTML = '\u03a3 a<sub>j</sub> = ' + sum.toFixed(2);

    state.attentionWeights.initialized = true;
  }

  const takeaway = document.getElementById('attn21-takeaway');
  if (takeaway) takeaway.textContent = ATTN_WGT_TAKEAWAYS[state.attentionWeights.step] || ATTN_WGT_TAKEAWAYS[0];
}

function setAttentionWeightsStep(step) {
  const slide = document.getElementById('slide-21');
  const takeaway = document.getElementById('attn21-takeaway');
  if (!slide) return;

  const clamped = Math.max(0, Math.min(ATTN_WGT_MAX_STEP, step));
  state.attentionWeights.step = clamped;

  slide.classList.toggle('attn21-show-scale', clamped >= 1);
  slide.classList.toggle('attn21-show-softmax', clamped >= 2);
  slide.classList.toggle('attn21-show-final', clamped >= 3);
  if (takeaway) takeaway.textContent = ATTN_WGT_TAKEAWAYS[clamped] || ATTN_WGT_TAKEAWAYS[0];
}

function runAttentionWeightsStep() {
  if (!state.attentionWeights.initialized) initAttentionWeightsSlide();
  if (state.attentionWeights.step >= ATTN_WGT_MAX_STEP) return false;
  setAttentionWeightsStep(state.attentionWeights.step + 1);
  return true;
}

function resetAttentionWeightsSlide() {
  const slide = document.getElementById('slide-21');
  if (!slide) return;
  setAttentionWeightsStep(0);
}
