function createAttentionPositionMiniChip(token) {
  return createEl('div', { className: 'attn19-token-chip', text: token });
}

function createAttentionPositionContextPanel(kind) {
  if (kind === 'rnn') {
    const panel = createEl('div', {
      className: 'attn25-mini-panel attn25-rnn-panel',
      id: 'attn25-rnn-panel'
    });
    panel.appendChild(createEl('div', {
      className: 'attn25-panel-title',
      text: 'RNN'
    }));
    const flow = createEl('div', { className: 'attn25-rnn-flow' });
    ATTN_POS_SEQ_A.forEach((token, idx) => {
      flow.appendChild(createAttentionPositionMiniChip(token));
      if (idx < ATTN_POS_SEQ_A.length - 1) {
        flow.appendChild(createEl('span', {
          className: 'attn25-rnn-arrow',
          text: '→'
        }));
      }
    });
    panel.appendChild(flow);
    panel.appendChild(createEl('div', {
      className: 'attn25-panel-note',
      text: 'one token at a time'
    }));
    return panel;
  }

  const panel = createEl('div', {
    className: 'attn25-mini-panel attn25-transformer-panel',
    id: 'attn25-transformer-panel'
  });
  panel.appendChild(createEl('div', {
    className: 'attn25-panel-title',
    text: 'Transformer'
  }));
  const flow = createEl('div', { className: 'attn25-transformer-flow' });
  const chips = createEl('div', { className: 'attn25-chip-inline' });
  ATTN_POS_SEQ_A.forEach((token) => {
    chips.appendChild(createAttentionPositionMiniChip(token));
  });
  flow.appendChild(chips);
  flow.appendChild(createEl('div', {
    className: 'arch-box attn',
    text: 'Self-Attention'
  }));
  panel.appendChild(flow);
  panel.appendChild(createEl('div', {
    className: 'attn25-panel-note',
    text: 'all tokens together'
  }));
  return panel;
}

function createAttentionPositionSeqStrip(kind) {
  const tokens = kind === 'a' ? ATTN_POS_SEQ_A : ATTN_POS_SEQ_B;
  return createEl('div', {
    className: 'attn25-seq-strip',
    id: 'attn25-seq-strip-' + kind
  }, [
    createEl('div', {
      className: 'attn25-seq-label',
      text: kind === 'a' ? 'Sequence A' : 'Sequence B'
    }),
    createEl('div', {
      className: 'attn25-chip-inline'
    }, tokens.map((token) => createAttentionPositionMiniChip(token))),
    createEl('div', {
      className: 'attn25-collapse-arrow',
      id: 'attn25-collapse-arrow-' + kind,
      text: '↓'
    })
  ]);
}

function createAttentionPositionBagRow(token, rowIndex) {
  return createEl('div', {
    className: 'attn25-bag-row',
    id: 'attn25-bag-row-' + rowIndex,
    dataset: { token }
  }, [
    createEl('div', { className: 'attn19-token-chip', text: token }),
    createEl('div', {
      className: 'attn25-bag-row-vector'
    }, createAttentionSkeletonVectorRect('attn25-bag-vector-' + rowIndex, '', 4))
  ]);
}

function createAttentionPositionProblemStage() {
  const bagShell = createEl('div', {
    className: 'attn25-bag-shell',
    id: 'attn25-bag-shell'
  });
  bagShell.appendChild(createEl('div', {
    className: 'attn25-bag-slots',
    id: 'attn25-bag-slots'
  }, ['cat', 'sat', 'on', 'mat'].map((token, rowIndex) => createAttentionPositionBagRow(token, rowIndex))));

  return createEl('div', {
    className: 'attn25-problem-stage',
    id: 'attn25-problem-stage'
  }, [
    createEl('div', {
      className: 'attn25-seq-top',
      id: 'attn25-seq-top'
    }, [
      createAttentionPositionSeqStrip('a'),
      createAttentionPositionSeqStrip('b')
    ]),
    createEl('div', {
      className: 'attn25-problem-body',
      id: 'attn25-problem-body'
    }, [
      createEl('div', {
        className: 'attn25-bag-wrap',
        id: 'attn25-bag-wrap'
      }, [
        createEl('div', {
          className: 'attn25-bag-label',
          id: 'attn25-bag-label',
          text: 'Token Embeddings Only'
        }),
        createEl('div', {
          className: 'attn25-bag-subtitle',
          id: 'attn25-bag-subtitle',
          text: 'same bag of token rows'
        }),
        bagShell
      ]),
      createEl('div', {
        className: 'attn25-problem-attn',
        id: 'attn25-problem-attn'
      }, createEl('div', {
        className: 'arch-box attn',
        text: 'Self-Attention'
      }))
    ]),
    createEl('div', {
      className: 'attn25-note-row',
      id: 'attn25-problem-notes'
    }, [
      createEl('div', {
        className: 'attn25-shared-note',
        id: 'attn25-note-bag',
        text: 'same tokens + no positions = same bag'
      }),
      createEl('div', {
        className: 'attn25-shared-note',
        id: 'attn25-note-orderblind',
        text: 'swap rows in → swap rows out'
      })
    ])
  ]);
}

function createAttentionPositionFixRow(token, rowIndex, sequenceKind) {
  return createEl('div', {
    className: 'attn25-fix-row',
    id: 'attn25-fix-row-' + sequenceKind + '-' + rowIndex,
    dataset: {
      token,
      posIndex: String(rowIndex + 1),
      compareToken: token === 'cat' || token === 'mat' ? 'true' : 'false'
    }
  }, [
    createEl('div', { className: 'attn19-token-chip', text: token }),
    createEl('div', {
      className: 'attn25-fix-row-vector'
    }, createAttentionSkeletonVectorRect('attn25-fix-vector-' + sequenceKind + '-' + rowIndex, '', 4))
  ]);
}

function createAttentionPositionFixCard(kind) {
  const tokens = kind === 'a' ? ATTN_POS_SEQ_A : ATTN_POS_SEQ_B;
  return createEl('div', {
    className: 'attn25-fix-card',
    id: 'attn25-fix-card-' + kind,
    dataset: { kind }
  }, [
    createEl('div', {
      className: 'attn25-fix-header'
    }, [
      createEl('div', {
        className: 'attn25-seq-label',
        text: kind === 'a' ? 'Sequence A' : 'Sequence B'
      }),
      createEl('div', {
        className: 'attn25-panel-note',
        text: tokens.join(' ')
      })
    ]),
    createEl('div', {
      className: 'attn25-pos-source',
      id: 'attn25-pos-source-' + kind
    }, createEl('div', {
      className: 'arch-box pos',
      text: 'Positional Embeddings'
    })),
    createEl('div', {
      className: 'attn25-fix-flow',
      id: 'attn25-fix-flow-' + kind
    }, [
      createEl('div', {
        className: 'attn25-pos-col',
        id: 'attn25-pos-col-' + kind
      }, ATTN_POS_TAGS.map((tag, rowIndex) => createEl('div', {
        className: 'attn25-pos-row',
        dataset: { posIndex: String(rowIndex + 1) },
        html: inlineMath('p_' + (rowIndex + 1))
      }))),
      createEl('div', {
        className: 'attn25-plus-col',
        id: 'attn25-plus-col-' + kind
      }, ATTN_POS_TAGS.map((_, rowIndex) => createEl('div', {
        className: 'attn25-plus-row',
        id: 'attn25-plus-row-' + kind + '-' + rowIndex
      }, createEl('div', {
        className: 'arch-plus',
        text: '+'
      })))),
      createEl('div', {
        className: 'attn25-fix-input-wrap',
        id: 'attn25-fix-input-wrap-' + kind
      }, createEl('div', {
        className: 'attn25-fix-input-shell',
        id: 'attn25-fix-input-shell-' + kind
      }, createEl('div', {
        className: 'attn25-fix-input-slots',
        id: 'attn25-fix-input-slots-' + kind
      }, tokens.map((token, rowIndex) => createAttentionPositionFixRow(token, rowIndex, kind))))),
      createEl('div', {
        className: 'attn25-fix-attn-wrap',
        id: 'attn25-fix-attn-' + kind
      }, createEl('div', {
        className: 'arch-box attn',
        text: 'Self-Attention'
      }))
    ])
  ]);
}

function createAttentionPositionSharedNotes() {
  return createEl('div', {
    className: 'attn25-note-row',
    id: 'attn25-fix-notes'
  }, [
    createEl('div', {
      className: 'attn25-shared-note',
      id: 'attn25-note-formula',
      html: inlineMath("x'_i = x_i + p_i")
    }),
    createEl('div', {
      className: 'attn25-shared-note',
      id: 'attn25-note-diff',
      html: inlineMath('\\mathrm{cat} + p_1 \\neq \\mathrm{cat} + p_4')
    }),
    createEl('div', {
      className: 'attn25-shared-note',
      id: 'attn25-note-saviour',
      text: 'Positional embeddings to the rescue'
    })
  ]);
}

function initAttentionPositionSlide() {
  const slide = document.getElementById('slide-25');
  const stage = document.getElementById('attn25-stage');
  const takeaway = document.getElementById('attn25-takeaway');
  if (!slide || !stage || !takeaway) return;

  if (!state.attentionPosition.initialized) {
    stage.innerHTML = '';
    stage.appendChild(createEl('div', {
      className: 'attn25-context-row',
      id: 'attn25-context-row'
    }, [
      createAttentionPositionContextPanel('rnn'),
      createAttentionPositionContextPanel('transformer')
    ]));
    stage.appendChild(createEl('div', {
      className: 'attn25-question-chip',
      id: 'attn25-question-chip',
      text: 'Where is the sequence info?'
    }));
    stage.appendChild(createAttentionPositionProblemStage());
    stage.appendChild(createEl('div', {
      className: 'attn25-fix-stage',
      id: 'attn25-fix-stage'
    }, [
      createAttentionPositionFixCard('a'),
      createAttentionPositionFixCard('b'),
      createAttentionPositionSharedNotes()
    ]));
    state.attentionPosition.initialized = true;
  }

  takeaway.innerHTML = ATTN_POS_TAKEAWAYS[state.attentionPosition.step] || ATTN_POS_TAKEAWAYS[0];
  typesetMath(slide);
}

function setAttentionPositionStep(step) {
  const slide = document.getElementById('slide-25');
  const takeaway = document.getElementById('attn25-takeaway');
  if (!slide || !takeaway) return;

  const clamped = Math.max(0, Math.min(ATTN_POS_MAX_STEP, step));
  state.attentionPosition.step = clamped;
  takeaway.innerHTML = ATTN_POS_TAKEAWAYS[clamped] || ATTN_POS_TAKEAWAYS[0];

  slide.classList.toggle('attn25-show-question', clamped >= 1);
  slide.classList.toggle('attn25-show-problem', clamped >= 2);
  slide.classList.toggle('attn25-show-fix', clamped >= 3);
}

function runAttentionPositionStep() {
  if (!state.attentionPosition.initialized) initAttentionPositionSlide();
  if (state.attentionPosition.step >= ATTN_POS_MAX_STEP) return false;
  setAttentionPositionStep(state.attentionPosition.step + 1);
  return true;
}

function resetAttentionPositionSlide() {
  const slide = document.getElementById('slide-25');
  if (!slide) return;
  state.attentionPosition.timers.forEach((timerId) => clearTimeout(timerId));
  state.attentionPosition.timers = [];
  state.attentionPosition.rafIds.forEach((rafId) => cancelAnimationFrame(rafId));
  state.attentionPosition.rafIds = [];
  setAttentionPositionStep(0);
}

/* =====================================================
   Slide-19 (Step 1) — Q/K/V projection fan-out
   ===================================================== */
