"""
Language Modeling Forward Pass Animation
=========================================
A reusable Manim animation showing the complete forward pass of a transformer
language model, from text input to next-token probability distribution.

Phase 1: Black-box view — sentence in, distribution out, token selected & appended (×2)
Phase 2: Smooth zoom into the black box
Phase 3: Detailed internal pipeline walkthrough

Render commands:
    manim -pqh pipeline.py LanguageModelingPipeline
    manim -pqk pipeline.py LanguageModelingPipeline  # 4K quality

Manim Community Edition v0.18+
"""

from manim import *
import numpy as np

# =============================================================================
# PARAMETERS
# =============================================================================

SENTENCE = "The cat sat on the"
TOKENS = ["The", "cat", "sat", "on", "the"]
TOKEN_IDS = [464, 3797, 3332, 319, 262]
T = 5  # sequence length
D_MODEL = 8  # embedding dimension (visual abstraction)
NUM_LAYERS = 12  # transformer layers
NUM_HEADS = 2  # attention heads (visual abstraction)
D_K = 4  # head dimension (D_MODEL // NUM_HEADS)
TOP_K = 5

# Autoregressive iterations for the black-box demonstration
BLACKBOX_ITERATIONS = [
    {
        "selected": "mat",
        "top_tokens": ["mat", "floor", "couch", "bed", "..."],
        "probs": [0.42, 0.18, 0.15, 0.12, 0.13],
    },
    {
        "selected": ".",
        "top_tokens": [".", ",", "and", "!", "..."],
        "probs": [0.55, 0.20, 0.10, 0.08, 0.07],
    },
]

# Visual constants
DARK_BG = "#1a1a2e"
ACCENT_COLOR = "#4fc3f7"
ACCENT_COLOR_2 = "#81c784"
ACCENT_COLOR_3 = "#ffb74d"
TEXT_COLOR = WHITE
BOX_COLOR = "#2d2d44"
LLM_BOX_FILL = "#0d0d1a"
MATRIX_COLORS = [BLUE_E, BLUE_D, BLUE_C, BLUE_B, BLUE_A, GREEN_A, YELLOW_A, ORANGE]
MASK_NEG_INF_COLOR = "#e57373"
MASK_PASS_COLOR = "#81c784"


# =============================================================================
# REUSABLE CLASSES
# =============================================================================

class LLMBlackBox(VGroup):
    """Opaque 'black box' representation of an LLM."""

    def __init__(self, width=2.8, height=1.8, **kwargs):
        super().__init__(**kwargs)

        self.box = RoundedRectangle(
            width=width, height=height, corner_radius=0.2,
            stroke_color=ACCENT_COLOR, stroke_width=3,
            fill_color=LLM_BOX_FILL, fill_opacity=0.95,
        )
        self.title_text = Text("LLM", font_size=36, color=ACCENT_COLOR, weight=BOLD)
        self.title_text.move_to(self.box.get_center() + UP * 0.15)

        # self.subtitle = Text("Language Model", font_size=13, color=GRAY_B)
        # self.subtitle.next_to(self.title_text, DOWN, buff=0.12)

        self.add(self.box, self.title_text)


class CompactDistribution(VGroup):
    """Compact horizontal-bar probability distribution."""

    def __init__(self, tokens, probs, bar_max_width=1.4, **kwargs):
        super().__init__(**kwargs)
        self.token_groups = VGroup()
        max_prob = max(probs)

        # First pass: create rows
        rows = []
        for i, (tok, prob) in enumerate(zip(tokens, probs)):
            label = Text(tok, font_size=16, color=TEXT_COLOR)
            bar = Rectangle(
                width=max((prob / max_prob) * bar_max_width, 0.06),
                height=0.22,
                fill_color=ACCENT_COLOR, fill_opacity=0.7,
                stroke_width=0,
            )
            pct = Text(f"{prob:.0%}", font_size=12, color=GRAY_B)
            label.shift(DOWN * i * 0.38)
            rows.append((label, bar, pct))

        # Align all bars to start at the same x
        max_label_right = max(r[0].get_right()[0] for r in rows)
        bar_x = max_label_right + 0.25

        for label, bar, pct in rows:
            bar.move_to(
                np.array([bar_x + bar.width / 2, label.get_center()[1], 0])
            )
            pct.next_to(bar, RIGHT, buff=0.1)
            row_group = VGroup(label, bar, pct)
            self.token_groups.add(row_group)

        self.add(self.token_groups)
        self.center()


class TokenRow(VGroup):
    """Row of token boxes."""

    def __init__(self, tokens, box_width=1.2, box_height=0.6, spacing=0.15, **kwargs):
        super().__init__(**kwargs)
        self.tokens = tokens
        self.boxes = VGroup()
        self.labels = VGroup()

        for i, token in enumerate(tokens):
            box = RoundedRectangle(
                width=box_width, height=box_height,
                corner_radius=0.1, stroke_color=ACCENT_COLOR, stroke_width=2,
                fill_color=BOX_COLOR, fill_opacity=0.8,
            )
            label = Text(f'"{token}"', font_size=20, color=TEXT_COLOR)
            label.move_to(box.get_center())
            token_group = VGroup(box, label)
            token_group.shift(RIGHT * i * (box_width + spacing))
            self.boxes.add(box)
            self.labels.add(label)
            self.add(token_group)
        self.center()


class IdRow(VGroup):
    """Row of integer token-ID boxes."""

    def __init__(self, ids, box_width=1.2, box_height=0.5, spacing=0.15, **kwargs):
        super().__init__(**kwargs)
        self.ids = ids
        self.boxes = VGroup()
        self.labels = VGroup()

        for i, token_id in enumerate(ids):
            box = RoundedRectangle(
                width=box_width, height=box_height,
                corner_radius=0.1, stroke_color=ACCENT_COLOR_2, stroke_width=2,
                fill_color=BOX_COLOR, fill_opacity=0.8,
            )
            label = Text(str(token_id), font_size=18, color=ACCENT_COLOR_2)
            label.move_to(box.get_center())
            id_group = VGroup(box, label)
            id_group.shift(RIGHT * i * (box_width + spacing))
            self.boxes.add(box)
            self.labels.add(label)
            self.add(id_group)
        self.center()


class TensorMatrix(VGroup):
    """Abstract heatmap grid representing a tensor."""

    def __init__(self, rows, cols, cell_size=0.25,
                 color_gradient=True, label=None, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols
        self.cells = VGroup()

        np.random.seed(42)
        for i in range(rows):
            for j in range(cols):
                if color_gradient:
                    idx = min(
                        int((i + j) / (rows + cols) * len(MATRIX_COLORS)),
                        len(MATRIX_COLORS) - 1,
                    )
                    fill_color = MATRIX_COLORS[idx]
                    opacity = 0.3 + 0.5 * np.random.random()
                else:
                    fill_color = BLUE_D
                    opacity = 0.5
                cell = Rectangle(
                    width=cell_size, height=cell_size,
                    stroke_color=WHITE, stroke_width=0.5,
                    fill_color=fill_color, fill_opacity=opacity,
                )
                cell.move_to(RIGHT * j * cell_size + DOWN * i * cell_size)
                self.cells.add(cell)

        self.add(self.cells)
        self.cells.center()

        self.border = Rectangle(
            width=cols * cell_size + 0.05,
            height=rows * cell_size + 0.05,
            stroke_color=WHITE, stroke_width=2, fill_opacity=0,
        )
        self.border.move_to(self.cells.get_center())
        self.add(self.border)

        if label:
            self.label = Text(label, font_size=16, color=GRAY_B)
            self.label.next_to(self, DOWN, buff=0.15)
            self.add(self.label)

    def get_cell(self, row, col):
        return self.cells[row * self.cols + col]


class TransformerBlockIcon(VGroup):
    """Collapsed transformer-block stack with ×N badge."""

    def __init__(self, num_layers=12, width=1.5, height=0.8, **kwargs):
        super().__init__(**kwargs)
        self.layers = VGroup()
        for i in range(3):
            offset = i * 0.08
            box = RoundedRectangle(
                width=width, height=height, corner_radius=0.1,
                stroke_color=ACCENT_COLOR_3, stroke_width=2,
                fill_color=BOX_COLOR, fill_opacity=0.9 - i * 0.2,
            )
            box.shift(UP * offset + LEFT * offset * 0.5)
            self.layers.add(box)
        self.add(self.layers)

        self.block_label = Text("Transformer", font_size=16, color=TEXT_COLOR)
        self.block_label.move_to(self.layers[0].get_center() + UP * 0.1)
        self.add(self.block_label)

        self.count_label = Text(f"×{num_layers}", font_size=20, color=ACCENT_COLOR_3)
        self.count_label.next_to(self.layers[0], DOWN, buff=0.1)
        self.add(self.count_label)


class ExpandedTransformerBlock(VGroup):
    """Detailed single-block diagram: LN→Attn→Add, LN→MLP→Add."""

    def __init__(self, width=2.5, **kwargs):
        super().__init__(**kwargs)
        self.components = VGroup()
        self.arrows = VGroup()

        component_data = [
            ("LayerNorm", BLUE_D),
            ("Attention", ACCENT_COLOR),
            ("+ (Residual)", ACCENT_COLOR_2),
            ("LayerNorm", BLUE_D),
            ("MLP", ACCENT_COLOR_3),
            ("+ (Residual)", ACCENT_COLOR_2),
        ]
        box_h = 0.45
        sp = 0.6

        for i, (name, color) in enumerate(component_data):
            box = RoundedRectangle(
                width=width, height=box_h, corner_radius=0.08,
                stroke_color=color, stroke_width=2,
                fill_color=BOX_COLOR, fill_opacity=0.85,
            )
            box.shift(DOWN * i * sp)
            label = Text(name, font_size=14, color=color)
            label.move_to(box.get_center())
            self.components.add(VGroup(box, label))

        self.add(self.components)

        for i in range(len(component_data) - 1):
            arrow = Arrow(
                self.components[i].get_bottom(),
                self.components[i + 1].get_top(),
                buff=0.05, stroke_width=2, color=GRAY_B,
                max_tip_length_to_length_ratio=0.15,
            )
            self.arrows.add(arrow)
        self.add(self.arrows)

        self.input_arrow = Arrow(
            self.components[0].get_top() + UP * 0.3,
            self.components[0].get_top(),
            buff=0.05, stroke_width=2, color=WHITE,
            max_tip_length_to_length_ratio=0.2,
        )
        self.output_arrow = Arrow(
            self.components[-1].get_bottom(),
            self.components[-1].get_bottom() + DOWN * 0.3,
            buff=0.05, stroke_width=2, color=WHITE,
            max_tip_length_to_length_ratio=0.2,
        )
        self.add(self.input_arrow, self.output_arrow)
        self.center()

        # Residual skip-connection arcs
        self.residual_arrows = VGroup()
        res1 = CurvedArrow(
            self.components[0].get_left() + LEFT * 0.1,
            self.components[2].get_left() + LEFT * 0.1,
            angle=-TAU / 4, color=ACCENT_COLOR_2, stroke_width=1.5,
        )
        res2 = CurvedArrow(
            self.components[3].get_left() + LEFT * 0.1,
            self.components[5].get_left() + LEFT * 0.1,
            angle=-TAU / 4, color=ACCENT_COLOR_2, stroke_width=1.5,
        )
        self.residual_arrows.add(res1, res2)
        self.add(self.residual_arrows)


class EquationLabel(VGroup):
    """LaTeX equation with optional subtitle."""

    def __init__(self, equation_tex, description=None, font_size=28, **kwargs):
        super().__init__(**kwargs)
        self.equation = MathTex(equation_tex, font_size=font_size, color=TEXT_COLOR)
        self.add(self.equation)
        if description:
            self.description = Text(description, font_size=16, color=GRAY_B)
            self.description.next_to(self.equation, DOWN, buff=0.15)
            self.add(self.description)


class CausalMaskMatrix(VGroup):
    """Triangular causal mask matrix with -inf / 0 coloring."""

    def __init__(self, size=T, cell_size=0.35, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.cells = VGroup()
        self.cell_labels = VGroup()

        for i in range(size):
            for j in range(size):
                is_masked = j > i
                fill_color = MASK_NEG_INF_COLOR if is_masked else MASK_PASS_COLOR
                fill_opacity = 0.6 if is_masked else 0.3
                cell = Rectangle(
                    width=cell_size, height=cell_size,
                    stroke_color=WHITE, stroke_width=0.5,
                    fill_color=fill_color, fill_opacity=fill_opacity,
                )
                cell.move_to(RIGHT * j * cell_size + DOWN * i * cell_size)
                self.cells.add(cell)

                text = r"-\infty" if is_masked else "0"
                lbl = MathTex(text, font_size=10, color=WHITE)
                lbl.move_to(cell.get_center())
                self.cell_labels.add(lbl)

        self.add(self.cells, self.cell_labels)
        self.center()

        self.border = Rectangle(
            width=size * cell_size + 0.05,
            height=size * cell_size + 0.05,
            stroke_color=WHITE, stroke_width=2, fill_opacity=0,
        )
        self.border.move_to(self.cells.get_center())
        self.add(self.border)

    def get_cell(self, row, col):
        return self.cells[row * self.size + col]


class AttentionScoreGrid(VGroup):
    """T x T attention score heatmap with programmable cell coloring."""

    def __init__(self, size=T, cell_size=0.35, label=None, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.cells = VGroup()

        for i in range(size):
            for j in range(size):
                cell = Rectangle(
                    width=cell_size, height=cell_size,
                    stroke_color=WHITE, stroke_width=0.5,
                    fill_color=BLUE_E, fill_opacity=0.2,
                )
                cell.move_to(RIGHT * j * cell_size + DOWN * i * cell_size)
                self.cells.add(cell)

        self.add(self.cells)
        self.cells.center()

        self.border = Rectangle(
            width=size * cell_size + 0.05,
            height=size * cell_size + 0.05,
            stroke_color=WHITE, stroke_width=2, fill_opacity=0,
        )
        self.border.move_to(self.cells.get_center())
        self.add(self.border)

        if label:
            self.label_text = Text(label, font_size=16, color=GRAY_B)
            self.label_text.next_to(self, DOWN, buff=0.15)
            self.add(self.label_text)

    def get_cell(self, row, col):
        return self.cells[row * self.size + col]


class ProbabilityBarChart(VGroup):
    """Vertical bar chart of token probabilities."""

    def __init__(self, tokens, probs, width=4, height=2, **kwargs):
        super().__init__(**kwargs)
        self.bars = VGroup()
        self.labels = VGroup()
        bar_width = width / len(tokens) * 0.7
        spacing = width / len(tokens)
        max_prob = max(probs)

        for i, (token, prob) in enumerate(zip(tokens, probs)):
            bar_height = (prob / max_prob) * height
            bar = Rectangle(
                width=bar_width, height=bar_height,
                fill_color=ACCENT_COLOR, fill_opacity=0.8,
                stroke_color=WHITE, stroke_width=1,
            )
            bar.move_to(
                RIGHT * (i * spacing - width / 2 + spacing / 2) + UP * bar_height / 2
            )
            self.bars.add(bar)

            token_label = Text(token, font_size=14, color=TEXT_COLOR)
            token_label.next_to(bar, DOWN, buff=0.1)
            self.labels.add(token_label)

            prob_label = Text(f"{prob:.2f}", font_size=12, color=GRAY_B)
            prob_label.next_to(bar, UP, buff=0.05)
            self.labels.add(prob_label)

        self.add(self.bars, self.labels)
        self.center()


# =============================================================================
# MAIN SCENE
# =============================================================================

class LanguageModelingPipeline(Scene):
    """Full animation: black-box demo → zoom-in → detailed pipeline."""

    def construct(self):
        self.camera.background_color = DARK_BG

        # Phase 1 — Black-box autoregressive demo
        self.show_black_box_overview()

        # Phase 2 — Zoom into the black box
        self.show_zoom_transition()

        # Phase 3 — Detailed internal pipeline
        self.show_tokenization()
        self.show_token_ids()
        self.show_embeddings()
        self.show_positional_embeddings()
        self.show_transformer_block()
        self.show_lm_head()
        self.show_softmax()
        self.show_distribution()
        self.show_ending()

    # ------------------------------------------------------------------
    # PHASE 1 — Black-box overview
    # ------------------------------------------------------------------

    def show_black_box_overview(self):
        """Title, LLM black box, and two autoregressive iterations."""

        # Title
        title = Text(
            "How Language Models Generate Text",
            font_size=30, color=ACCENT_COLOR,
        )
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)

        # LLM black box
        llm_box = LLMBlackBox()
        llm_box.move_to(ORIGIN)
        self.play(FadeIn(llm_box, scale=0.85), run_time=1)

        # Sentence to the left
        current_text = SENTENCE
        sentence = Text(f'"{current_text}"', font_size=20, color=TEXT_COLOR)
        sentence.next_to(llm_box, LEFT, buff=1.6)

        arrow_in = Arrow(
            sentence.get_right(), llm_box.get_left(),
            buff=0.2, color=GRAY_B, stroke_width=2,
        )
        self.play(FadeIn(sentence, shift=RIGHT * 0.3), GrowArrow(arrow_in), run_time=1)
        self.wait(0.5)

        # Store for later
        self.bb_title = title
        self.bb_box = llm_box
        self.bb_sentence = sentence
        self.bb_arrow_in = arrow_in

        # Run two autoregressive iterations
        for iteration in BLACKBOX_ITERATIONS:
            current_text = self._run_iteration(iteration, current_text)

    def _run_iteration(self, data, current_text):
        """One autoregressive step: process → distribution → select → append."""
        llm = self.bb_box

        # --- processing pulse ---
        self.play(
            llm.box.animate.set_stroke(YELLOW, width=5),
            run_time=0.25,
        )
        self.play(
            llm.box.animate.set_stroke(ACCENT_COLOR, width=3),
            run_time=0.25,
        )

        # --- output arrow ---
        arrow_out = Arrow(
            llm.get_right(), llm.get_right() + RIGHT * 0.9,
            buff=0.15, color=GRAY_B, stroke_width=2,
        )
        self.play(GrowArrow(arrow_out), run_time=0.4)

        # --- compact distribution ---
        dist = CompactDistribution(data["top_tokens"], data["probs"])
        dist.next_to(arrow_out, RIGHT, buff=0.25)
        self.play(
            LaggedStart(
                *[FadeIn(r, shift=LEFT * 0.2) for r in dist.token_groups],
                lag_ratio=0.08,
            ),
            run_time=0.8,
        )

        # --- highlight selected token ---
        sel_idx = data["top_tokens"].index(data["selected"])
        highlight = SurroundingRectangle(
            dist.token_groups[sel_idx],
            color=YELLOW, stroke_width=2, buff=0.06,
        )
        self.play(Create(highlight), run_time=0.4)
        self.wait(0.4)

        # --- flying token arc back to sentence ---
        selected = data["selected"]
        flying = Text(selected, font_size=22, color=YELLOW, weight=BOLD)
        flying.move_to(dist.token_groups[sel_idx][0])

        # Target: right edge of current sentence
        target = self.bb_sentence.get_right() + RIGHT * 0.25 + UP * 0.0
        self.play(FadeIn(flying, scale=1.2), run_time=0.2)
        self.play(
            flying.animate(path_arc=-PI / 3).move_to(target),
            run_time=0.9,
            rate_func=smooth,
        )

        # --- update sentence ---
        if selected in {".", ",", "!", "?", ";", ":"}:
            new_text = current_text + selected
        else:
            new_text = current_text + " " + selected

        new_sentence = Text(f'"{new_text}"', font_size=20, color=TEXT_COLOR)
        new_sentence.move_to(self.bb_sentence.get_center())

        # Rebuild arrow to match new sentence position
        new_arrow_in = Arrow(
            new_sentence.get_right(), self.bb_box.get_left(),
            buff=0.2, color=GRAY_B, stroke_width=2,
        )

        self.play(
            FadeOut(flying),
            FadeOut(dist), FadeOut(highlight), FadeOut(arrow_out),
            Transform(self.bb_sentence, new_sentence),
            Transform(self.bb_arrow_in, new_arrow_in),
            run_time=0.7,
        )
        self.wait(0.3)
        return new_text

    # ------------------------------------------------------------------
    # PHASE 2 — Zoom transition
    # ------------------------------------------------------------------

    def show_zoom_transition(self):
        """Animate 'entering' the LLM black box to reveal internals."""

        # Question prompt
        question = Text("But how does this work?", font_size=28, color=ACCENT_COLOR)
        question.next_to(self.bb_box, DOWN, buff=0.9)
        self.play(Write(question), run_time=1.2)
        self.wait(0.8)

        # Fade everything except the LLM box
        self.play(
            FadeOut(self.bb_title),
            FadeOut(self.bb_sentence),
            FadeOut(self.bb_arrow_in),
            FadeOut(question),
            run_time=0.8,
        )

        # Centre the box
        self.play(self.bb_box.animate.move_to(ORIGIN), run_time=0.4)

        # Pulsing glow
        self.play(
            self.bb_box.box.animate.set_stroke(YELLOW, width=6),
            rate_func=there_and_back,
            run_time=0.8,
        )

        # Fade the label text inside the box
        self.play(
            self.bb_box.title_text.animate.set_opacity(0),
            # self.bb_box.subtitle.animate.set_opacity(0),
            run_time=0.5,
        )

        # Create a dark cover that will expand from the box outline
        zoom_cover = self.bb_box.box.copy()
        zoom_cover.set_fill(DARK_BG, opacity=1)
        zoom_cover.set_stroke(ACCENT_COLOR, width=3)
        self.add(zoom_cover)
        self.remove(self.bb_box)

        # Expand cover to fill the entire viewport
        self.play(
            zoom_cover.animate.scale(6).set_stroke(width=0),
            run_time=2.0,
            rate_func=rush_into,
        )

        # --- Screen is now fully dark --- prepare Phase 3 content ---

        inner_title = Text(
            "Inside the Language Model",
            font_size=32, color=ACCENT_COLOR,
        )
        inner_title.to_edge(UP, buff=0.5)

        sentence_label = Text("Input:", font_size=20, color=GRAY_B)
        sentence_text = Text(f'"{SENTENCE}"', font_size=28, color=TEXT_COLOR)
        sentence_group = VGroup(sentence_label, sentence_text).arrange(RIGHT, buff=0.3)
        sentence_group.next_to(inner_title, DOWN, buff=0.8)

        # Place behind the cover (invisible for now)
        self.add_behind(inner_title, sentence_label, sentence_text)

        # Reveal by fading the cover away
        self.play(
            FadeOut(zoom_cover),
            run_time=1.5,
            rate_func=rush_from,
        )

        # Store references used by Phase 3 methods
        self.title = inner_title
        self.sentence_text = sentence_text
        self.sentence_label = sentence_label

    # helper — add mobjects behind all current mobjects
    def add_behind(self, *mobjects):
        for m in mobjects:
            self.add(m)
            self.bring_to_back(m)

    # ------------------------------------------------------------------
    # PHASE 3 — Detailed pipeline
    # ------------------------------------------------------------------

    def show_tokenization(self):
        """Text → token boxes."""
        stage_label = Text("1. Tokenization", font_size=24, color=ACCENT_COLOR_2)
        stage_label.to_edge(LEFT, buff=0.5).shift(UP * 2)

        token_row = TokenRow(TOKENS)
        token_row.next_to(self.sentence_text, DOWN, buff=1)

        arrow = Arrow(
            self.sentence_text.get_bottom(), token_row.get_top(),
            buff=0.2, color=GRAY_B, stroke_width=2,
        )

        eq = EquationLabel(
            r"\text{text} \rightarrow [\text{tok}_1, \ldots, \text{tok}_T]"
        )
        eq.scale(0.7).to_edge(RIGHT, buff=0.8).shift(UP * 1.5)

        self.play(FadeIn(stage_label))
        self.play(GrowArrow(arrow), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(b, scale=0.8) for b in token_row], lag_ratio=0.1),
            run_time=1.5,
        )
        self.play(Write(eq), run_time=0.8)
        self.wait(0.5)

        self.token_row = token_row
        self.token_arrow = arrow
        self.stage_label = stage_label
        self.current_eq = eq

    def show_token_ids(self):
        """Tokens → integer IDs."""
        new_stage = Text("2. Token IDs", font_size=24, color=ACCENT_COLOR_2)
        new_stage.move_to(self.stage_label)

        id_row = IdRow(TOKEN_IDS)
        id_row.next_to(self.token_row, DOWN, buff=0.8)

        arrows = VGroup(*[
            Arrow(
                self.token_row.boxes[i].get_bottom(), id_row.boxes[i].get_top(),
                buff=0.1, color=GRAY_B, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.2,
            )
            for i in range(len(TOKENS))
        ])

        new_eq = EquationLabel(r"\text{tok}_i \rightarrow \text{id}_i \in \mathbb{Z}")
        new_eq.scale(0.7).move_to(self.current_eq)

        self.play(Transform(self.stage_label, new_stage))
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[FadeIn(b, scale=0.8) for b in id_row], lag_ratio=0.1),
            Transform(self.current_eq, new_eq),
            run_time=1.2,
        )
        self.wait(0.3)

        self.id_row = id_row
        self.id_arrows = arrows

    def show_embeddings(self):
        """IDs → embedding-matrix lookup."""
        self.play(
            FadeOut(self.title), FadeOut(self.sentence_text),
            FadeOut(self.sentence_label), FadeOut(self.token_arrow),
            self.token_row.animate.scale(0.7).to_edge(UP, buff=0.4).shift(LEFT * 2),
            self.id_row.animate.scale(0.7).to_edge(UP, buff=1.2).shift(LEFT * 2),
            FadeOut(self.id_arrows),
            run_time=1,
        )

        new_stage = Text("3. Embedding Lookup", font_size=24, color=ACCENT_COLOR_2)
        new_stage.move_to(self.stage_label)

        emb_matrix = TensorMatrix(T, D_MODEL, cell_size=0.22, label=f"E: {T}×{D_MODEL}")
        emb_matrix.next_to(self.id_row, DOWN, buff=0.8)

        new_eq = EquationLabel(
            r"E = \text{Embed}[\text{ids}] \in \mathbb{R}^{T \times d}"
        )
        new_eq.scale(0.7).move_to(self.current_eq)

        self.play(Transform(self.stage_label, new_stage))
        self.play(
            FadeIn(emb_matrix, scale=0.9),
            Transform(self.current_eq, new_eq),
            run_time=1.5,
        )

        for i in range(T):
            hl = VGroup(*[
                emb_matrix.get_cell(i, j).copy().set_fill(YELLOW, opacity=0.6)
                for j in range(D_MODEL)
            ])
            self.play(FadeIn(hl), run_time=0.15)
            self.play(FadeOut(hl), run_time=0.1)

        self.wait(0.3)
        self.emb_matrix = emb_matrix

    def show_positional_embeddings(self):
        """Add positional encoding: X0 = E + P."""
        new_stage = Text("4. Add Positional Encoding", font_size=24, color=ACCENT_COLOR_2)
        new_stage.move_to(self.stage_label)

        pos_matrix = TensorMatrix(T, D_MODEL, cell_size=0.22, label=f"P: {T}×{D_MODEL}")
        pos_matrix.next_to(self.emb_matrix, RIGHT, buff=1)

        plus = MathTex("+", font_size=40, color=ACCENT_COLOR_2)
        plus.move_to((self.emb_matrix.get_right() + pos_matrix.get_left()) / 2)

        result_matrix = TensorMatrix(
            T, D_MODEL, cell_size=0.22, label=f"X\u2080: {T}\u00d7{D_MODEL}"
        )
        result_matrix.next_to(pos_matrix, RIGHT, buff=1)

        equals = MathTex("=", font_size=40, color=WHITE)
        equals.move_to((pos_matrix.get_right() + result_matrix.get_left()) / 2)

        new_eq = EquationLabel(r"X_0 = E + P")
        new_eq.scale(0.7).move_to(self.current_eq)

        self.play(Transform(self.stage_label, new_stage))
        self.play(
            FadeIn(pos_matrix, shift=RIGHT), Write(plus),
            Transform(self.current_eq, new_eq),
            run_time=1,
        )
        self.play(Write(equals), FadeIn(result_matrix, scale=0.9), run_time=1)
        self.wait(0.5)

        self.pos_matrix = pos_matrix
        self.result_matrix = result_matrix
        self.plus = plus
        self.equals = equals

    def show_transformer_block(self):
        """Collapsed block → zoom in → data flow → zoom back out."""
        self.play(
            FadeOut(self.token_row), FadeOut(self.id_row),
            FadeOut(self.emb_matrix), FadeOut(self.pos_matrix),
            FadeOut(self.plus), FadeOut(self.equals),
            self.result_matrix.animate.scale(0.8)
                .to_edge(LEFT, buff=0.5).shift(UP * 0.5),
            run_time=1,
        )

        new_stage = Text("5. Transformer Blocks", font_size=24, color=ACCENT_COLOR_2)
        new_stage.move_to(self.stage_label)

        tf_block = TransformerBlockIcon(NUM_LAYERS)
        tf_block.next_to(self.result_matrix, RIGHT, buff=1.5)

        arrow_in = Arrow(
            self.result_matrix.get_right(), tf_block.get_left(),
            buff=0.2, color=WHITE, stroke_width=2,
        )
        output_label = MathTex(r"X_N", font_size=28, color=TEXT_COLOR)
        output_label.next_to(tf_block, RIGHT, buff=1)
        arrow_out = Arrow(
            tf_block.get_right(), output_label.get_left(),
            buff=0.2, color=WHITE, stroke_width=2,
        )

        new_eq = EquationLabel(
            r"X_\ell = \text{TransformerBlock}(X_{\ell-1})"
        )
        new_eq.scale(0.7).move_to(self.current_eq)

        self.play(Transform(self.stage_label, new_stage))
        self.play(
            GrowArrow(arrow_in), FadeIn(tf_block, scale=0.9),
            Transform(self.current_eq, new_eq),
            run_time=1,
        )
        self.play(GrowArrow(arrow_out), Write(output_label), run_time=0.8)
        self.wait(0.5)

        # --- Zoom into one block ---
        zoom_text = Text("Let's look inside one block...", font_size=20, color=GRAY_B)
        zoom_text.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(zoom_text))
        self.wait(0.5)

        self.play(
            FadeOut(self.result_matrix), FadeOut(arrow_in),
            FadeOut(arrow_out), FadeOut(output_label),
            FadeOut(self.stage_label), FadeOut(self.current_eq),
            FadeOut(zoom_text),
            run_time=0.8,
        )

        expanded = ExpandedTransformerBlock()
        expanded.scale(1.2).move_to(ORIGIN)
        self.play(ReplacementTransform(tf_block, expanded), run_time=1.5)

        block_eq = MathTex(
            r"X' &= X + \text{Attn}(\text{LN}(X)) \\"
            r" X'' &= X' + \text{MLP}(\text{LN}(X'))",
            font_size=24,
        ).to_edge(RIGHT, buff=0.5)
        self.play(Write(block_eq), run_time=1.5)

        # Animate a flow dot through LayerNorm, then pause at Attention
        flow_dot = Dot(color=YELLOW, radius=0.08)
        flow_dot.move_to(expanded.input_arrow.get_start())
        self.play(FadeIn(flow_dot))

        # Component 0: LayerNorm
        comp0 = expanded.components[0]
        self.play(
            flow_dot.animate.move_to(comp0.get_center()),
            comp0[0].animate.set_fill(YELLOW, opacity=0.3),
            run_time=0.4,
        )
        self.play(comp0[0].animate.set_fill(BOX_COLOR, opacity=0.85), run_time=0.2)

        # Component 1: Attention — pause and zoom in
        comp1 = expanded.components[1]
        self.play(
            flow_dot.animate.move_to(comp1.get_center()),
            comp1[0].animate.set_fill(YELLOW, opacity=0.3),
            run_time=0.4,
        )

        # Pulse the Attention box
        self.play(
            comp1[0].animate.set_stroke(YELLOW, width=4),
            run_time=0.3,
        )
        self.play(
            comp1[0].animate.set_stroke(ACCENT_COLOR, width=2),
            run_time=0.3,
        )

        attn_zoom_text = Text(
            "Let's look inside attention...", font_size=20, color=GRAY_B,
        )
        attn_zoom_text.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(attn_zoom_text), run_time=0.5)
        self.wait(0.5)

        # FadeOut expanded block for the deep dive
        self.play(
            FadeOut(expanded), FadeOut(block_eq),
            FadeOut(flow_dot), FadeOut(attn_zoom_text),
            run_time=0.8,
        )

        # === ATTENTION DEEP DIVE ===
        self.show_attention_deep_dive()

        # === RETURN FROM DEEP DIVE ===
        # Recreate expanded block and resume flow dot
        expanded = ExpandedTransformerBlock()
        expanded.scale(1.2).move_to(ORIGIN)
        block_eq = MathTex(
            r"X' &= X + \text{Attn}(\text{LN}(X)) \\"
            r" X'' &= X' + \text{MLP}(\text{LN}(X'))",
            font_size=24,
        ).to_edge(RIGHT, buff=0.5)
        self.play(FadeIn(expanded), FadeIn(block_eq), run_time=0.8)

        # Resume flow dot at Attention, continue through remaining components
        flow_dot = Dot(color=YELLOW, radius=0.08)
        flow_dot.move_to(expanded.components[1].get_center())
        self.play(FadeIn(flow_dot), run_time=0.2)
        self.play(
            expanded.components[1][0].animate.set_fill(BOX_COLOR, opacity=0.85),
            run_time=0.2,
        )

        # Components 2-5: Residual, LayerNorm, MLP, Residual
        for comp in expanded.components[2:]:
            self.play(
                flow_dot.animate.move_to(comp.get_center()),
                comp[0].animate.set_fill(YELLOW, opacity=0.3),
                run_time=0.4,
            )
            self.play(comp[0].animate.set_fill(BOX_COLOR, opacity=0.85), run_time=0.2)

        self.play(
            flow_dot.animate.move_to(expanded.output_arrow.get_end()),
            run_time=0.3,
        )
        self.play(FadeOut(flow_dot))
        self.wait(0.5)

        # --- Zoom back out ---
        tf_new = TransformerBlockIcon(NUM_LAYERS)
        tf_new.move_to(ORIGIN)
        self.play(
            FadeOut(block_eq),
            ReplacementTransform(expanded, tf_new),
            run_time=1,
        )
        self.play(tf_new.animate.shift(LEFT * 1.5), run_time=0.5)

        self.tf_block = tf_new

        self.output_matrix = TensorMatrix(
            T, D_MODEL, cell_size=0.2, label=f"X_N: {T}×{D_MODEL}"
        )
        self.output_matrix.next_to(self.tf_block, RIGHT, buff=1.5)
        arrow_out_new = Arrow(
            self.tf_block.get_right(), self.output_matrix.get_left(),
            buff=0.2, color=WHITE, stroke_width=2,
        )
        self.play(
            GrowArrow(arrow_out_new), FadeIn(self.output_matrix), run_time=0.8
        )
        self.arrow_to_output = arrow_out_new

    # ------------------------------------------------------------------
    # PHASE 3b — Attention deep dive
    # ------------------------------------------------------------------

    def show_attention_deep_dive(self):
        """Orchestrate the full attention mechanism walkthrough."""
        self.show_attn_pre_ln()
        self.show_attn_qkv_projections()
        self.show_attn_multihead_split()
        self.show_attn_score_computation()
        self.show_attn_softmax()
        self.show_attn_weighted_sum()
        self.show_attn_output_projection()
        self.show_attn_residual_add()

    def show_attn_pre_ln(self):
        """Step 5a: Pre-LN path — X → LayerNorm → LN(X)."""
        stage_label = Text("5a. Layer Norm (Pre-Attention)", font_size=24, color=ACCENT_COLOR_2)
        stage_label.to_edge(LEFT, buff=0.5).shift(UP * 2)

        # Input matrix X
        x_matrix = TensorMatrix(T, D_MODEL, cell_size=0.2, label=f"X: {T}×{D_MODEL}")
        x_matrix.move_to(LEFT * 4 + DOWN * 0.5)

        # LayerNorm box
        ln_box = RoundedRectangle(
            width=1.6, height=0.6, corner_radius=0.1,
            stroke_color=BLUE_D, stroke_width=2,
            fill_color=BOX_COLOR, fill_opacity=0.85,
        )
        ln_label = Text("LayerNorm", font_size=16, color=BLUE_D)
        ln_label.move_to(ln_box.get_center())
        ln_group = VGroup(ln_box, ln_label)
        ln_group.move_to(DOWN * 0.5)

        # Output LN(X)
        ln_output = TensorMatrix(T, D_MODEL, cell_size=0.2, label=f"LN(X): {T}×{D_MODEL}")
        ln_output.move_to(RIGHT * 4 + DOWN * 0.5)

        # Arrows
        arr1 = Arrow(
            x_matrix.get_right(), ln_group.get_left(),
            buff=0.2, color=GRAY_B, stroke_width=2,
        )
        arr2 = Arrow(
            ln_group.get_right(), ln_output.get_left(),
            buff=0.2, color=GRAY_B, stroke_width=2,
        )

        eq = EquationLabel(r"\hat{X} = \text{LayerNorm}(X)")
        eq.scale(0.7).to_edge(RIGHT, buff=0.8).shift(UP * 1.5)

        self.play(FadeIn(stage_label))
        self.play(FadeIn(x_matrix, scale=0.9), run_time=0.8)
        self.play(GrowArrow(arr1), FadeIn(ln_group, scale=0.9), run_time=0.8)
        self.play(
            ln_box.animate.set_fill(YELLOW, opacity=0.3),
            run_time=0.3,
        )
        self.play(
            ln_box.animate.set_fill(BOX_COLOR, opacity=0.85),
            run_time=0.3,
        )
        self.play(GrowArrow(arr2), FadeIn(ln_output, scale=0.9), Write(eq), run_time=1)
        self.wait(0.5)

        # Clean up stage visuals, keep X and LN(X)
        self.play(
            FadeOut(stage_label), FadeOut(eq),
            FadeOut(ln_group), FadeOut(arr1), FadeOut(arr2),
            run_time=0.6,
        )

        self.attn_input_x = x_matrix
        self.attn_ln_output = ln_output

    def show_attn_qkv_projections(self):
        """Step 5b: QKV projections — LN(X) → Q, K, V via weight matrices."""
        stage_label = Text("5b. QKV Projections", font_size=24, color=ACCENT_COLOR_2)
        stage_label.to_edge(LEFT, buff=0.5).shift(UP * 2)

        # Reposition input X to upper-left corner (small), keep LN(X) as source
        self.play(
            self.attn_input_x.animate.scale(0.6).to_corner(UL, buff=0.4),
            self.attn_ln_output.animate.move_to(LEFT * 4),
            run_time=0.8,
        )

        # Three branch targets
        branch_data = [
            ("Q", ACCENT_COLOR, r"W_Q"),
            ("K", ACCENT_COLOR_2, r"W_K"),
            ("V", ACCENT_COLOR_3, r"W_V"),
        ]

        matrices = []
        arrows = []
        weight_labels = []

        for i, (name, color, w_tex) in enumerate(branch_data):
            y_offset = (i - 1) * 2.0  # spread vertically: -2, 0, +2
            mat = TensorMatrix(
                T, D_MODEL, cell_size=0.18,
                label=f"{name}: {T}×{D_MODEL}",
            )
            mat.move_to(RIGHT * 3 + UP * y_offset)

            arr = Arrow(
                self.attn_ln_output.get_right(),
                mat.get_left(),
                buff=0.2, color=color, stroke_width=2,
            )
            w_label = MathTex(w_tex, font_size=18, color=color)
            w_label.next_to(arr, UP, buff=0.05)

            matrices.append(mat)
            arrows.append(arr)
            weight_labels.append(w_label)

        eq = EquationLabel(r"Q = XW_Q, \; K = XW_K, \; V = XW_V")
        eq.scale(0.7).to_edge(RIGHT, buff=0.5).shift(UP * 2)

        self.play(FadeIn(stage_label))
        self.play(
            LaggedStart(
                *[AnimationGroup(GrowArrow(a), Write(w))
                  for a, w in zip(arrows, weight_labels)],
                lag_ratio=0.2,
            ),
            run_time=1.2,
        )
        self.play(
            LaggedStart(
                *[FadeIn(m, scale=0.9) for m in matrices],
                lag_ratio=0.15,
            ),
            Write(eq),
            run_time=1.5,
        )
        self.wait(0.5)

        # Clean up source and arrows, keep Q, K, V
        self.play(
            FadeOut(stage_label), FadeOut(eq),
            FadeOut(self.attn_ln_output),
            *[FadeOut(a) for a in arrows],
            *[FadeOut(w) for w in weight_labels],
            run_time=0.6,
        )

        self.attn_q_matrix = matrices[0]
        self.attn_k_matrix = matrices[1]
        self.attn_v_matrix = matrices[2]

    def show_attn_multihead_split(self):
        """Step 5c: Multi-head split (schematic) — Q,K,V split into heads."""
        stage_label = Text("5c. Multi-Head Split", font_size=24, color=ACCENT_COLOR_2)
        stage_label.to_edge(LEFT, buff=0.5).shift(UP * 2)

        eq = EquationLabel(
            r"Q, K, V \in \mathbb{R}^{T \times d}"
            r"\rightarrow h \text{ heads of } \mathbb{R}^{T \times d_k}"
        )
        eq.scale(0.65).to_edge(RIGHT, buff=0.5).shift(UP * 2)

        self.play(FadeIn(stage_label), Write(eq), run_time=1)

        # For each matrix, show a bracket splitting into NUM_HEADS smaller copies
        split_groups = VGroup()
        original_matrices = [self.attn_q_matrix, self.attn_k_matrix, self.attn_v_matrix]
        names = ["Q", "K", "V"]
        colors = [ACCENT_COLOR, ACCENT_COLOR_2, ACCENT_COLOR_3]

        for mat, name, color in zip(original_matrices, names, colors):
            # Create head sub-matrices next to the original
            head_group = VGroup()
            for h in range(NUM_HEADS):
                head_mat = TensorMatrix(T, D_K, cell_size=0.15)
                head_label = Text(f"H{h+1}", font_size=12, color=color)
                head_label.next_to(head_mat, DOWN, buff=0.08)
                head_unit = VGroup(head_mat, head_label)
                head_group.add(head_unit)
            head_group.arrange(RIGHT, buff=0.3)
            head_group.move_to(mat.get_center())
            split_groups.add(head_group)

        # Animate split: fade original → show heads
        self.play(
            *[FadeOut(m) for m in original_matrices],
            *[FadeIn(sg, scale=0.9) for sg in split_groups],
            run_time=1.2,
        )

        note = Text("Each head attends independently", font_size=16, color=GRAY_B)
        note.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(0.8)

        # Morph back — indicate we'll show one head
        transition_text = Text(
            f"Showing one head (d_k = {D_K})", font_size=18, color=ACCENT_COLOR,
        )
        transition_text.to_edge(DOWN, buff=0.8)

        # Recreate Q, K, V at single-head size (T × D_K)
        q_head = TensorMatrix(T, D_K, cell_size=0.2, label=f"q: {T}×{D_K}")
        k_head = TensorMatrix(T, D_K, cell_size=0.2, label=f"k: {T}×{D_K}")
        v_head = TensorMatrix(T, D_K, cell_size=0.2, label=f"v: {T}×{D_K}")
        q_head.move_to(LEFT * 4 + UP * 2)
        k_head.move_to(LEFT * 4)
        v_head.move_to(LEFT * 4 + DOWN * 2)

        self.play(
            FadeOut(note),
            FadeIn(transition_text),
            *[FadeOut(sg) for sg in split_groups],
            run_time=0.6,
        )
        self.play(
            LaggedStart(
                FadeIn(q_head, scale=0.9),
                FadeIn(k_head, scale=0.9),
                FadeIn(v_head, scale=0.9),
                lag_ratio=0.1,
            ),
            run_time=1,
        )
        self.wait(0.5)

        self.play(
            FadeOut(stage_label), FadeOut(eq), FadeOut(transition_text),
            run_time=0.5,
        )

        self.attn_q_matrix = q_head
        self.attn_k_matrix = k_head
        self.attn_v_matrix = v_head

    def show_attn_score_computation(self):
        """Step 5d: Attention score computation — dot product, scaling, causal mask."""
        stage_label = Text("5d. Attention Scores", font_size=24, color=ACCENT_COLOR_2)
        stage_label.to_edge(LEFT, buff=0.5).shift(UP * 2)
        self.play(FadeIn(stage_label))

        # --- Phase 4a: Dot product zoom ---
        # Move Q to upper-left, K below it, hide V
        self.play(
            self.attn_q_matrix.animate.move_to(LEFT * 4 + UP * 1.5),
            self.attn_k_matrix.animate.move_to(LEFT * 4 + DOWN * 0.5),
            self.attn_v_matrix.animate.set_opacity(0.2).move_to(RIGHT * 5 + DOWN * 2),
            run_time=0.8,
        )

        # Highlight row i=2 of Q and row j=1 of K
        q_row_hl = VGroup(*[
            self.attn_q_matrix.get_cell(2, j).copy().set_fill(ACCENT_COLOR, opacity=0.6)
            for j in range(D_K)
        ])
        k_row_hl = VGroup(*[
            self.attn_k_matrix.get_cell(1, j).copy().set_fill(ACCENT_COLOR_2, opacity=0.6)
            for j in range(D_K)
        ])

        qi_label = MathTex(r"q_i", font_size=22, color=ACCENT_COLOR)
        qi_label.next_to(q_row_hl, RIGHT, buff=0.3)
        kj_label = MathTex(r"k_j", font_size=22, color=ACCENT_COLOR_2)
        kj_label.next_to(k_row_hl, RIGHT, buff=0.3)

        self.play(FadeIn(q_row_hl), FadeIn(qi_label), run_time=0.5)
        self.play(FadeIn(k_row_hl), FadeIn(kj_label), run_time=0.5)

        # Show the two vectors side by side for dot product
        q_vec = Rectangle(width=D_K * 0.3, height=0.35,
                          fill_color=ACCENT_COLOR, fill_opacity=0.5,
                          stroke_color=ACCENT_COLOR, stroke_width=2)
        k_vec = Rectangle(width=D_K * 0.3, height=0.35,
                          fill_color=ACCENT_COLOR_2, fill_opacity=0.5,
                          stroke_color=ACCENT_COLOR_2, stroke_width=2)
        q_vec.move_to(RIGHT * 0.5 + UP * 0.5)
        k_vec.move_to(RIGHT * 0.5 + DOWN * 0.5)

        q_vec_label = MathTex(r"q_i^T", font_size=20, color=ACCENT_COLOR)
        q_vec_label.next_to(q_vec, LEFT, buff=0.15)
        k_vec_label = MathTex(r"k_j", font_size=20, color=ACCENT_COLOR_2)
        k_vec_label.next_to(k_vec, LEFT, buff=0.15)

        dot_symbol = MathTex(r"\cdot", font_size=30, color=WHITE)
        dot_symbol.move_to((q_vec.get_center() + k_vec.get_center()) / 2)

        score_result = MathTex(r"= \text{score}_{i,j}", font_size=20, color=YELLOW)
        score_result.next_to(VGroup(q_vec, k_vec), RIGHT, buff=0.4)

        self.play(
            FadeIn(q_vec), FadeIn(k_vec),
            FadeIn(q_vec_label), FadeIn(k_vec_label),
            Write(dot_symbol),
            run_time=0.8,
        )

        eq1 = EquationLabel(r"\text{score}_{i,j} = q_i^T k_j")
        eq1.scale(0.7).to_edge(RIGHT, buff=0.8).shift(UP * 1.5)
        self.play(Write(eq1), FadeIn(score_result), run_time=0.8)
        self.wait(0.5)

        # --- Phase 4b: Scaling ---
        eq2 = EquationLabel(
            r"\text{score}_{i,j} = \frac{q_i^T k_j}{\sqrt{d_k}}"
        )
        eq2.scale(0.7).to_edge(RIGHT, buff=0.8).shift(UP * 1.5)

        scale_label = MathTex(
            r"\div \sqrt{" + str(D_K) + r"}", font_size=22, color=ACCENT_COLOR_3,
        )
        scale_label.next_to(score_result, DOWN, buff=0.2)

        self.play(Transform(eq1, eq2), FadeIn(scale_label), run_time=0.8)
        self.wait(0.5)

        # Clean dot product visuals
        self.play(
            FadeOut(q_row_hl), FadeOut(k_row_hl),
            FadeOut(qi_label), FadeOut(kj_label),
            FadeOut(q_vec), FadeOut(k_vec),
            FadeOut(q_vec_label), FadeOut(k_vec_label),
            FadeOut(dot_symbol), FadeOut(score_result), FadeOut(scale_label),
            FadeOut(eq1),
            run_time=0.6,
        )

        # --- Phase 4c: Full score matrix + causal mask ---
        # Show QK^T as a T×T grid
        score_grid = AttentionScoreGrid(T, cell_size=0.35, label="S (raw scores)")
        score_grid.move_to(LEFT * 2.5 + DOWN * 0.5)

        # Populate with pseudo-random intensities
        np.random.seed(123)
        for i in range(T):
            for j in range(T):
                val = np.random.random()
                score_grid.get_cell(i, j).set_fill(
                    ACCENT_COLOR, opacity=0.2 + 0.6 * val,
                )

        qkt_eq = EquationLabel(r"S = \frac{QK^T}{\sqrt{d_k}}")
        qkt_eq.scale(0.7).to_edge(RIGHT, buff=0.8).shift(UP * 1.5)

        # Fade Q and K matrices, show score grid
        self.play(
            FadeOut(self.attn_q_matrix), FadeOut(self.attn_k_matrix),
            FadeIn(score_grid, scale=0.9), Write(qkt_eq),
            run_time=1,
        )
        self.wait(0.3)

        # Show causal mask
        mask = CausalMaskMatrix(T, cell_size=0.35)
        mask_label = Text("M (causal mask)", font_size=16, color=GRAY_B)
        mask_label.next_to(mask, DOWN, buff=0.15)
        mask_group = VGroup(mask, mask_label)
        mask_group.move_to(RIGHT * 2.5 + DOWN * 0.5)

        plus_sign = MathTex("+", font_size=36, color=WHITE)
        plus_sign.move_to((score_grid.get_right() + mask.get_left()) / 2)

        self.play(
            FadeIn(mask_group, shift=RIGHT * 0.3),
            Write(plus_sign),
            run_time=1,
        )
        self.wait(0.3)

        # Animate mask being applied: upper-triangle cells in score_grid turn red
        mask_anims = []
        for i in range(T):
            for j in range(T):
                if j > i:
                    mask_anims.append(
                        score_grid.get_cell(i, j).animate.set_fill(
                            MASK_NEG_INF_COLOR, opacity=0.6,
                        )
                    )
        self.play(
            LaggedStart(*mask_anims, lag_ratio=0.03),
            run_time=1.2,
        )

        # Update equation
        full_eq = EquationLabel(r"S = \frac{QK^T}{\sqrt{d_k}} + M")
        full_eq.scale(0.7).to_edge(RIGHT, buff=0.8).shift(UP * 1.5)
        self.play(Transform(qkt_eq, full_eq), run_time=0.6)
        self.wait(0.5)

        # Clean up mask visual, keep score grid as the combined result
        self.play(
            FadeOut(mask_group), FadeOut(plus_sign),
            FadeOut(stage_label), FadeOut(qkt_eq),
            score_grid.animate.move_to(LEFT * 2 + DOWN * 0.3),
            run_time=0.8,
        )

        # Update label
        if hasattr(score_grid, 'label_text'):
            new_label = Text("S (masked scores)", font_size=16, color=GRAY_B)
            new_label.next_to(score_grid, DOWN, buff=0.15)
            self.play(Transform(score_grid.label_text, new_label), run_time=0.3)

        self.attn_score_matrix = score_grid

    def show_attn_softmax(self):
        """Step 5e: Row-wise softmax over attention scores → attention weights A."""
        stage_label = Text("5e. Attention Softmax", font_size=24, color=ACCENT_COLOR_2)
        stage_label.to_edge(LEFT, buff=0.5).shift(UP * 2)

        eq = EquationLabel(r"A = \text{softmax}(\text{rows of } S)")
        eq.scale(0.7).to_edge(RIGHT, buff=0.8).shift(UP * 1.5)

        self.play(FadeIn(stage_label), Write(eq), run_time=0.8)

        # Row-by-row softmax animation
        for i in range(T):
            row_cells = [self.attn_score_matrix.get_cell(i, j) for j in range(T)]

            # Highlight row
            row_hl = VGroup(*[
                c.copy().set_stroke(YELLOW, width=3) for c in row_cells
            ])
            self.play(FadeIn(row_hl), run_time=0.15)

            # Compute fake softmax probabilities (causal: j > i → 0)
            raw = []
            for j in range(T):
                if j > i:
                    raw.append(0.0)
                else:
                    raw.append(np.random.random() * 0.5 + 0.3)
            total = sum(raw)
            probs = [r / total if total > 0 else 0 for r in raw]

            # Color cells by probability
            color_anims = []
            for j in range(T):
                if j > i:
                    color_anims.append(
                        row_cells[j].animate.set_fill(
                            MASK_NEG_INF_COLOR, opacity=0.08,
                        )
                    )
                else:
                    color_anims.append(
                        row_cells[j].animate.set_fill(
                            ACCENT_COLOR, opacity=0.15 + 0.7 * probs[j],
                        )
                    )

            self.play(*color_anims, run_time=0.25)
            self.play(FadeOut(row_hl), run_time=0.1)

        note = Text("Each row sums to 1", font_size=16, color=GRAY_B)
        note.next_to(self.attn_score_matrix, RIGHT, buff=0.5)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(0.5)

        # Relabel
        if hasattr(self.attn_score_matrix, 'label_text'):
            a_label = Text("A (attention weights)", font_size=16, color=GRAY_B)
            a_label.next_to(self.attn_score_matrix, DOWN, buff=0.15)
            self.play(Transform(self.attn_score_matrix.label_text, a_label), run_time=0.3)

        self.play(FadeOut(stage_label), FadeOut(eq), FadeOut(note), run_time=0.5)

        self.attn_weights_matrix = self.attn_score_matrix

    def show_attn_weighted_sum(self):
        """Step 5f: Weighted value sum — O = A · V."""
        stage_label = Text("5f. Weighted Value Sum", font_size=24, color=ACCENT_COLOR_2)
        stage_label.to_edge(LEFT, buff=0.5).shift(UP * 2)

        eq = EquationLabel(r"O = A \cdot V")
        eq.scale(0.7).to_edge(RIGHT, buff=0.8).shift(UP * 1.5)

        # Bring V back
        self.play(
            self.attn_v_matrix.animate.set_opacity(1).move_to(RIGHT * 0.5 + DOWN * 0.3),
            run_time=0.8,
        )

        # Position: A (left) × V (middle) = O (right)
        times_sign = MathTex(r"\times", font_size=30, color=WHITE)
        times_sign.move_to(
            (self.attn_weights_matrix.get_right() + self.attn_v_matrix.get_left()) / 2
        )

        o_matrix = TensorMatrix(T, D_K, cell_size=0.2, label=f"O: {T}×{D_K}")
        o_matrix.move_to(RIGHT * 4 + DOWN * 0.3)

        equals_sign = MathTex("=", font_size=30, color=WHITE)
        equals_sign.move_to(
            (self.attn_v_matrix.get_right() + o_matrix.get_left()) / 2
        )

        self.play(FadeIn(stage_label), Write(eq), run_time=0.8)
        self.play(Write(times_sign), run_time=0.3)

        # Show weighted combination for one row
        demo_row = 2
        row_cells = [self.attn_weights_matrix.get_cell(demo_row, j) for j in range(T)]
        row_hl = VGroup(*[c.copy().set_stroke(YELLOW, width=3) for c in row_cells])
        self.play(FadeIn(row_hl), run_time=0.3)

        # Highlight V rows with varying opacity (simulating weight)
        v_highlights = VGroup()
        for j in range(T):
            weight_opacity = row_cells[j].get_fill_opacity()
            for k in range(D_K):
                cell_hl = self.attn_v_matrix.get_cell(j, k).copy()
                cell_hl.set_fill(YELLOW, opacity=weight_opacity * 0.8)
                v_highlights.add(cell_hl)

        self.play(FadeIn(v_highlights), run_time=0.5)
        self.wait(0.3)
        self.play(FadeOut(row_hl), FadeOut(v_highlights), run_time=0.3)

        # Show result
        self.play(
            Write(equals_sign),
            FadeIn(o_matrix, scale=0.9),
            run_time=1,
        )
        self.wait(0.5)

        # Clean up
        self.play(
            FadeOut(stage_label), FadeOut(eq),
            FadeOut(self.attn_weights_matrix), FadeOut(self.attn_v_matrix),
            FadeOut(times_sign), FadeOut(equals_sign),
            o_matrix.animate.move_to(LEFT * 2),
            run_time=0.8,
        )

        self.attn_output_o = o_matrix

    def show_attn_output_projection(self):
        """Step 5g: Output projection — Attn(X) = Concat(heads) · W_O."""
        stage_label = Text("5g. Output Projection", font_size=24, color=ACCENT_COLOR_2)
        stage_label.to_edge(LEFT, buff=0.5).shift(UP * 2)

        # Arrow from O to result via W_O
        result_matrix = TensorMatrix(
            T, D_MODEL, cell_size=0.2,
            label=f"Attn(X): {T}×{D_MODEL}",
        )
        result_matrix.move_to(RIGHT * 3)

        arrow = Arrow(
            self.attn_output_o.get_right(), result_matrix.get_left(),
            buff=0.2, color=WHITE, stroke_width=2,
        )
        wo_label = MathTex(r"W_O", font_size=20, color=ACCENT_COLOR_3)
        wo_label.next_to(arrow, UP, buff=0.1)

        eq = EquationLabel(r"\text{Attn}(X) = \text{Concat}(\text{heads}) \cdot W_O")
        eq.scale(0.65).to_edge(RIGHT, buff=0.5).shift(UP * 2)

        note = Text("In full model: concat all heads, then project", font_size=16, color=GRAY_B)
        note.to_edge(DOWN, buff=0.8)

        self.play(FadeIn(stage_label), run_time=0.5)
        self.play(
            GrowArrow(arrow), Write(wo_label),
            FadeIn(result_matrix, shift=RIGHT * 0.3),
            Write(eq),
            run_time=1.2,
        )
        self.play(FadeIn(note), run_time=0.4)
        self.wait(0.5)

        # Clean up
        self.play(
            FadeOut(stage_label), FadeOut(eq), FadeOut(note),
            FadeOut(self.attn_output_o), FadeOut(arrow), FadeOut(wo_label),
            run_time=0.6,
        )

        self.attn_result = result_matrix

    def show_attn_residual_add(self):
        """Step 5h: Residual connection — X' = X + Attn(LN(X))."""
        stage_label = Text("5h. Residual Connection", font_size=24, color=ACCENT_COLOR_2)
        stage_label.to_edge(LEFT, buff=0.5).shift(UP * 2)

        # Bring X back to normal size and position
        self.play(
            self.attn_input_x.animate.scale(1 / 0.6).move_to(LEFT * 4 + DOWN * 0.3),
            self.attn_result.animate.move_to(ORIGIN + DOWN * 0.3),
            run_time=0.8,
        )

        # Plus sign
        plus_sign = MathTex("+", font_size=36, color=ACCENT_COLOR_2)
        plus_sign.move_to(
            (self.attn_input_x.get_right() + self.attn_result.get_left()) / 2
        )

        # Result X'
        x_prime = TensorMatrix(
            T, D_MODEL, cell_size=0.2, label=f"X': {T}×{D_MODEL}",
        )
        x_prime.move_to(RIGHT * 4 + DOWN * 0.3)

        equals_sign = MathTex("=", font_size=36, color=WHITE)
        equals_sign.move_to(
            (self.attn_result.get_right() + x_prime.get_left()) / 2
        )

        # Residual arc
        res_arc = CurvedArrow(
            self.attn_input_x.get_top() + UP * 0.2,
            plus_sign.get_top() + UP * 0.2,
            angle=-TAU / 6, color=ACCENT_COLOR_2, stroke_width=1.5,
        )

        eq = EquationLabel(r"X' = X + \text{Attn}(\text{LN}(X))")
        eq.scale(0.7).to_edge(RIGHT, buff=0.5).shift(UP * 2)

        self.play(FadeIn(stage_label))
        self.play(Write(plus_sign), FadeIn(res_arc), run_time=0.8)
        self.play(Write(equals_sign), FadeIn(x_prime, scale=0.9), Write(eq), run_time=1)
        self.wait(0.8)

        # Final cleanup — clear everything
        self.play(
            FadeOut(stage_label), FadeOut(eq),
            FadeOut(self.attn_input_x), FadeOut(self.attn_result),
            FadeOut(plus_sign), FadeOut(equals_sign),
            FadeOut(x_prime), FadeOut(res_arc),
            run_time=0.8,
        )

    # ------------------------------------------------------------------
    # PHASE 3 continued — Post-transformer steps
    # ------------------------------------------------------------------

    def show_lm_head(self):
        """LM head projection: logits = X_N · W_U."""
        stage_label = Text("6. LM Head", font_size=24, color=ACCENT_COLOR_2)
        stage_label.to_edge(LEFT, buff=0.5).shift(UP * 2)

        logits_matrix = TensorMatrix(T, 6, cell_size=0.2, label="logits: T×V")
        logits_matrix.next_to(self.output_matrix, RIGHT, buff=1.5)

        arrow = Arrow(
            self.output_matrix.get_right(), logits_matrix.get_left(),
            buff=0.2, color=WHITE, stroke_width=2,
        )
        wu_label = MathTex(r"W_U", font_size=20, color=ACCENT_COLOR_3)
        wu_label.next_to(arrow, UP, buff=0.1)

        eq = EquationLabel(r"\text{logits} = X_N \cdot W_U")
        eq.scale(0.7).to_edge(RIGHT, buff=0.5).shift(UP * 2)

        self.play(FadeIn(stage_label))
        self.play(
            GrowArrow(arrow), Write(wu_label),
            FadeIn(logits_matrix, shift=RIGHT), Write(eq),
            run_time=1.5,
        )
        self.wait(0.5)

        self.stage_label = stage_label
        self.current_eq = eq
        self.logits_matrix = logits_matrix
        self.arrow_to_logits = arrow
        self.wu_label = wu_label

    def show_softmax(self):
        """Softmax over the last position's logits."""
        new_stage = Text("7. Softmax", font_size=24, color=ACCENT_COLOR_2)
        new_stage.move_to(self.stage_label)

        softmax_box = RoundedRectangle(
            width=1.2, height=0.6, corner_radius=0.1,
            stroke_color=ACCENT_COLOR, stroke_width=2,
            fill_color=BOX_COLOR, fill_opacity=0.8,
        )
        softmax_label = MathTex(r"\sigma", font_size=28, color=ACCENT_COLOR)
        softmax_label.move_to(softmax_box)
        softmax = VGroup(softmax_box, softmax_label)
        softmax.next_to(self.logits_matrix, DOWN, buff=0.8)

        arr1 = Arrow(
            self.logits_matrix.get_bottom(), softmax.get_top(),
            buff=0.15, color=WHITE, stroke_width=2,
        )
        prob_label = MathTex(r"p \in [0,1]^V", font_size=22, color=TEXT_COLOR)
        prob_label.next_to(softmax, DOWN, buff=0.4)
        arr2 = Arrow(
            softmax.get_bottom(), prob_label.get_top(),
            buff=0.1, color=WHITE, stroke_width=2,
        )

        new_eq = EquationLabel(r"p = \text{softmax}(\text{logits}_{[-1]})")
        new_eq.scale(0.7).move_to(self.current_eq)

        self.play(Transform(self.stage_label, new_stage))
        self.play(
            GrowArrow(arr1), FadeIn(softmax, scale=0.9),
            Transform(self.current_eq, new_eq),
            run_time=1,
        )
        self.play(GrowArrow(arr2), Write(prob_label), run_time=0.8)
        self.wait(0.5)

        self.softmax = softmax
        self.prob_label = prob_label

    def show_distribution(self):
        """Top-k bar chart of next-token probabilities."""
        self.play(
            FadeOut(self.tf_block), FadeOut(self.output_matrix),
            FadeOut(self.arrow_to_output), FadeOut(self.logits_matrix),
            FadeOut(self.arrow_to_logits), FadeOut(self.wu_label),
            FadeOut(self.softmax), FadeOut(self.prob_label),
            FadeOut(self.stage_label), FadeOut(self.current_eq),
            run_time=0.8,
        )

        stage = Text("8. Next Token Distribution", font_size=28, color=ACCENT_COLOR_2)
        stage.to_edge(UP, buff=0.5)

        top_tokens = ["mat", "floor", "couch", "table", "bed"]
        top_probs = [0.42, 0.18, 0.15, 0.12, 0.08]

        bar_chart = ProbabilityBarChart(top_tokens, top_probs, width=5, height=2.5)
        bar_chart.move_to(ORIGIN)

        eq = EquationLabel(r"P(\text{next token} \mid \text{context})")
        eq.next_to(bar_chart, UP, buff=0.8)

        context = Text(f'Context: "{SENTENCE}"', font_size=18, color=GRAY_B)
        context.next_to(bar_chart, DOWN, buff=0.6)

        self.play(Write(stage), run_time=0.8)
        self.play(Write(eq), run_time=0.8)
        self.play(FadeIn(context))

        bar_chart.bars.save_state()
        for bar in bar_chart.bars:
            bar.stretch(0.01, 1, about_edge=DOWN)
        self.play(FadeIn(bar_chart.labels))
        self.play(Restore(bar_chart.bars), run_time=1.5)

        highlight = SurroundingRectangle(
            bar_chart.bars[0], color=YELLOW, stroke_width=2, buff=0.05
        )
        self.play(Create(highlight), run_time=0.5)
        self.wait(1)

        self.bar_chart = bar_chart
        self.highlight = highlight
        self.stage_label_final = stage
        self.eq_final = eq
        self.context_text = context

    def show_ending(self):
        """End card."""
        self.play(
            FadeOut(self.highlight),
            self.bar_chart.animate.set_opacity(0.5),
            run_time=0.5,
        )
        end_text = Text(
            "We'll zoom into each part later...",
            font_size=24, color=ACCENT_COLOR,
        )
        end_text.to_edge(DOWN, buff=1)
        self.play(Write(end_text), run_time=1)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


# =============================================================================
# QUICK-TEST SCENE
# =============================================================================

class QuickTest(LanguageModelingPipeline):
    """Quick-test scene that reuses all LanguageModelingPipeline methods.

    Uncomment whichever phase/step you want to test.
    Methods that depend on earlier state need their prerequisites uncommented too.

    Dependency chain:
        show_black_box_overview      (standalone)
        show_zoom_transition         (needs: show_black_box_overview)
        show_tokenization            (needs: show_zoom_transition)
        show_token_ids               (needs: show_tokenization)
        show_embeddings              (needs: show_token_ids)
        show_positional_embeddings   (needs: show_embeddings)
        show_transformer_block       (needs: show_positional_embeddings)
          |-- calls show_attention_deep_dive internally, which runs:
          |     |-- show_attn_pre_ln
          |     |-- show_attn_qkv_projections    (needs: show_attn_pre_ln)
          |     |-- show_attn_multihead_split    (needs: show_attn_qkv_projections)
          |     |-- show_attn_score_computation  (needs: show_attn_multihead_split)
          |     |-- show_attn_softmax            (needs: show_attn_score_computation)
          |     |-- show_attn_weighted_sum       (needs: show_attn_softmax)
          |     |-- show_attn_output_projection  (needs: show_attn_weighted_sum)
          |     |-- show_attn_residual_add       (needs: show_attn_output_projection)
        show_lm_head                 (needs: show_transformer_block)
        show_softmax                 (needs: show_lm_head)
        show_distribution            (needs: show_softmax)
        show_ending                  (needs: show_distribution)
    """

    def construct(self):
        self.camera.background_color = DARK_BG

        # --- Phase 1: Black-box autoregressive demo ---
        # self.show_black_box_overview()

        # --- Phase 2: Zoom into the black box ---
        # self.show_zoom_transition()

        # --- Phase 3: Detailed internal pipeline ---
        # self.show_tokenization()
        # self.show_token_ids()
        # self.show_embeddings()
        # self.show_positional_embeddings()
        # self.show_transformer_block()

        # --- Phase 3b: Attention deep dive (standalone test) ---
        # self.show_attn_pre_ln()
        # self.show_attn_qkv_projections()
        # self.show_attn_multihead_split()
        # self.show_attn_score_computation()
        # self.show_attn_softmax()
        # self.show_attn_weighted_sum()
        # self.show_attn_output_projection()
        # self.show_attn_residual_add()

        # --- Phase 3 continued ---
        # self.show_lm_head()
        # self.show_softmax()
        # self.show_distribution()
        # self.show_ending()

        self.wait(1)
