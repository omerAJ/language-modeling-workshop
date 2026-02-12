"""
Language Modeling Forward Pass Animation (Simplified)
=====================================================
A reusable Manim animation showing the forward pass of a transformer
language model up to positional embeddings.

Render commands:
    manim -pqh pipeline.py LanguageModelingPipeline
    manim -pql pipeline.py LanguageModelingPipeline  # fast preview

Manim Community Edition v0.18+
"""

from manim import *
import numpy as np

# Fix font kerning issues (e.g., 'self', 'masked' rendering poorly)
Text.set_default(font="DejaVu Sans")

# =============================================================================
# PARAMETERS
# =============================================================================

SENTENCE = "The cat sat on the"
TOKENS = ["The", "cat", "sat", "on", "the"]
TOKEN_IDS = [464, 3797, 3332, 319, 262]
T = 5  # sequence length
D_MODEL = 8  # visual cell count for the d dimension (labels show "d")

wrap_width = config.frame_width * 0.85

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
TEXT_COLOR = WHITE
BOX_COLOR = "#2d2d44"
LLM_BOX_FILL = "#0d0d1a"
MATRIX_COLORS = [BLUE_E, BLUE_D, BLUE_C, BLUE_B, BLUE_A, GREEN_A, YELLOW_A, ORANGE]
EQUATION_BG = "#101624"
EQUATION_STROKE = "#fdd835"
TRANSFORMER_RED = "#e57373"
SLOW_FACTOR = 1.2


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clamp_to_frame(mobj, buff=0.2):
    """Ensures mobj stays fully inside the screen bounds.

    Parameters:
        mobj : Mobject  – The object to clamp.
        buff : float    – Margin from screen edges.
    """
    left_bound = -config.frame_x_radius + buff
    right_bound = config.frame_x_radius - buff
    bottom_bound = -config.frame_y_radius + buff
    top_bound = config.frame_y_radius - buff

    if mobj.get_left()[0] < left_bound:
        mobj.shift((left_bound - mobj.get_left()[0]) * RIGHT)
    if mobj.get_right()[0] > right_bound:
        mobj.shift((right_bound - mobj.get_right()[0]) * LEFT)
    if mobj.get_bottom()[1] < bottom_bound:
        mobj.shift((bottom_bound - mobj.get_bottom()[1]) * UP)
    if mobj.get_top()[1] > top_bound:
        mobj.shift((top_bound - mobj.get_top()[1]) * DOWN)


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
        self.theta_text = MathTex(r"\theta", font_size=32, color=ACCENT_COLOR)
        self.title_text.move_to(self.box.get_center() + UP * 0.3)
        self.theta_text.next_to(self.title_text, DOWN*1.2, buff=0.05)

        self.add(self.box, self.title_text, self.theta_text)


class CompactDistribution(VGroup):
    """Compact horizontal-bar probability distribution."""

    def __init__(self, tokens, probs, bar_max_width=1.4, **kwargs):
        super().__init__(**kwargs)
        self.token_groups = VGroup()
        max_prob = max(probs)

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

        max_label_right = max(r[0].get_right()[0] for r in rows)
        bar_x = max_label_right + 0.25

        for label, bar, pct in rows:
            bar.move_to(np.array([bar_x + bar.width / 2, label.get_center()[1], 0]))
            pct.next_to(bar, RIGHT, buff=0.1)
            self.token_groups.add(VGroup(label, bar, pct))

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

    def __init__(self, rows, cols, cell_size=0.25, label=None, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols
        self.cells = VGroup()

        np.random.seed(seed)
        for i in range(rows):
            for j in range(cols):
                idx = min(
                    int((i + j) / (rows + cols) * len(MATRIX_COLORS)),
                    len(MATRIX_COLORS) - 1,
                )
                fill_color = MATRIX_COLORS[idx]
                opacity = 0.3 + 0.5 * np.random.random()
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


class EmbeddingWeightMatrix(VGroup):
    """Embedding weight matrix W_E ∈ ℝ^{|V|×d} with ellipsis to show scale."""

    def __init__(self, num_cols=8, cell_size=0.22, top_rows=3, bottom_rows=2, **kwargs):
        super().__init__(**kwargs)
        self.num_cols = num_cols
        self.cell_size = cell_size

        np.random.seed(99)

        # Top visible rows
        self.top_section = VGroup(*[self._make_row(i) for i in range(top_rows)])
        self.top_section.arrange(DOWN, buff=0)

        # Ellipsis dots spread across the row width
        ellipsis_dots = VGroup()
        for _ in range(min(num_cols, 4)):
            d = MathTex(r"\vdots", font_size=12, color=GRAY_B)
            ellipsis_dots.add(d)
        ellipsis_dots.arrange(RIGHT, buff=0.3)
        self.ellipsis = ellipsis_dots

        # Bottom visible rows
        self.bottom_section = VGroup(*[self._make_row(50 + i) for i in range(bottom_rows)])
        self.bottom_section.arrange(DOWN, buff=0)

        # Assemble vertically
        content = VGroup(self.top_section, ellipsis_dots, self.bottom_section)
        content.arrange(DOWN, buff=0.1)
        self.content = content
        self.add(content)

        # Border
        self.border = SurroundingRectangle(
            content, buff=0.04,
            stroke_color=WHITE, stroke_width=2, fill_opacity=0,
        )
        self.add(self.border)

    def _make_row(self, seed):
        row = VGroup()
        for j in range(self.num_cols):
            idx = (seed * 3 + j * 7) % len(MATRIX_COLORS)
            fill_color = MATRIX_COLORS[idx]
            opacity = 0.3 + 0.5 * np.random.random()
            cell = Rectangle(
                width=self.cell_size, height=self.cell_size,
                stroke_color=WHITE, stroke_width=0.5,
                fill_color=fill_color, fill_opacity=opacity,
            )
            row.add(cell)
        row.arrange(RIGHT, buff=0)
        return row

    def make_extracted_row(self, seed):
        """Create a new row of cells representing an extracted embedding vector."""
        rng = np.random.RandomState(seed + 200)
        row = VGroup()
        for j in range(self.num_cols):
            idx = (seed * 5 + j * 3 + 2) % len(MATRIX_COLORS)
            fill_color = MATRIX_COLORS[idx]
            opacity = 0.35 + 0.45 * rng.random()
            cell = Rectangle(
                width=self.cell_size, height=self.cell_size,
                stroke_color=WHITE, stroke_width=0.5,
                fill_color=fill_color, fill_opacity=opacity,
            )
            row.add(cell)
        row.arrange(RIGHT, buff=0)
        return row


class EquationLabel(VGroup):
    """LaTeX equation with optional subtitle."""

    def __init__(self, equation_tex, description=None, font_size=35, **kwargs):
        super().__init__(**kwargs)
        if isinstance(equation_tex, (list, tuple)):
            equations = VGroup(*[
                MathTex(eq, font_size=font_size, color=TEXT_COLOR)
                for eq in equation_tex
            ])
            equations.arrange(DOWN, buff=0.12)
        else:
            equations = VGroup(
                MathTex(equation_tex, font_size=font_size, color=TEXT_COLOR)
            )

        self.equations = equations
        self.equation = equations

        content = equations
        if description:
            self.description = Text(description, font_size=20, color=GRAY_B)
            content = VGroup(equations, self.description).arrange(DOWN, buff=0.15)

        background = SurroundingRectangle(
            content,
            buff=0.2,
            corner_radius=0.12,
            stroke_color=EQUATION_STROKE,
            stroke_width=2,
            fill_color=EQUATION_BG,
            fill_opacity=0.85,
        )

        self.background = background
        self.add(background, content)


class PipelineStageBlock(VGroup):
    """A titled block with sub-item labels for the three-stage pipeline view."""

    def __init__(self, title, sub_items, color, width=3.4, height=2.8,
                 title_font_size=18, sub_font_size=14, **kwargs):
        super().__init__(**kwargs)

        self.box = RoundedRectangle(
            width=width, height=height, corner_radius=0.15,
            stroke_color=color, stroke_width=3,
            fill_color=BOX_COLOR, fill_opacity=0.85,
        )

        self.title_text = Text(
            title, font_size=title_font_size,
            color=color, weight=BOLD,
        )
        self.title_text.move_to(self.box.get_top() + DOWN * 0.35)

        line_left = self.box.get_left()[0] + 0.2
        line_right = self.box.get_right()[0] - 0.2
        line_y = self.title_text.get_bottom()[1] - 0.15
        self.separator = Line(
            np.array([line_left, line_y, 0]),
            np.array([line_right, line_y, 0]),
            stroke_color=color, stroke_width=1, stroke_opacity=0.5,
        )

        self.sub_labels = VGroup()
        start_y = line_y - 0.3
        for i, item in enumerate(sub_items):
            bullet = Text(f"  {item}", font_size=sub_font_size, color=GRAY_A)
            bullet.move_to(
                np.array([self.box.get_center()[0], start_y - i * 0.35, 0])
            )
            self.sub_labels.add(bullet)

        self.add(self.box, self.title_text, self.separator, self.sub_labels)


# =============================================================================
# MAIN SCENE
# =============================================================================

class LanguageModelingPipeline(Scene):
    """Black-box demo → zoom-in → tokenization → embeddings → positional embeddings."""

    def play(self, *args, **kwargs):
        run_time = kwargs.pop("run_time", 1.0)
        kwargs["run_time"] = run_time * SLOW_FACTOR
        return super().play(*args, **kwargs)

    def wait(self, duration=1.0):
        return super().wait(duration * SLOW_FACTOR)

    def construct(self):
        self.camera.background_color = DARK_BG

        self.show_black_box_overview()
        self.transition_to_three_stage()
        self.show_three_stage_pipeline()

        # Zoom into each stage
        self.transition_to_input_encoding()
        self.show_tokenization()
        self.show_token_ids()
        self.show_embeddings()
        self.show_positional_encoding()

        # Handoff: X₀ travels from Input Encoding → Transformer Core
        self.zoom_back_to_three_stage(
            handoff_from='input',
            handoff_to='transformer',
            matrix_label="X_0",
            narrative_text="We have successfully converted our text to a list of numbers... Now lets pass it to the transformer and see what it does to these!",
        )
        self.transition_to_transformer_core()
        self.show_transformer_pipeline()

        # Handoff: X_N travels from Transformer Core → Output Decoding
        self.zoom_back_to_three_stage(
            handoff_from='transformer',
            handoff_to='output',
            matrix_label="X_N",
            narrative_text="",
        )
        self.transition_to_output_decoding()
        self.show_output_decoding()
        self.wait(2)

    # ------------------------------------------------------------------
    # PHASE 1 — Black-box overview
    # ------------------------------------------------------------------

    def show_black_box_overview(self):
        """Title, LLM black box, and two autoregressive iterations."""

        title = Text(
            "How Language Models Generate Text",
            font_size=30, color=ACCENT_COLOR,
        )
        title.to_edge(UP, buff=0.5)
        clamp_to_frame(title)
        self.play(Write(title), run_time=1)

        llm_box = LLMBlackBox()
        llm_box.move_to(ORIGIN)
        self.play(FadeIn(llm_box, scale=0.85), run_time=1)

        ar_eq = EquationLabel(
            r"p(x)=\prod_{t=1}^{T} p(x_t\mid x_{<t};\theta)",
            font_size=24,
        )
        ar_eq.next_to(llm_box, UP, buff=0.35)
        clamp_to_frame(ar_eq)
        self.play(FadeIn(ar_eq), run_time=0.8)

        current_text = SENTENCE
        sentence = Text(f'"{current_text}"', font_size=20, color=TEXT_COLOR)
        sentence.next_to(llm_box, LEFT, buff=1.6)
        clamp_to_frame(sentence)

        arrow_in = Arrow(
            sentence.get_right(), llm_box.get_left(),
            buff=0.2, color=GRAY_B, stroke_width=2,
        )
        self.play(FadeIn(sentence, shift=RIGHT * 0.3), GrowArrow(arrow_in), run_time=1)
        self.wait(0.5)

        self.bb_title = title
        self.bb_box = llm_box
        self.bb_sentence = sentence
        self.bb_arrow_in = arrow_in
        self.bb_eq = ar_eq

        for iteration in BLACKBOX_ITERATIONS:
            current_text = self._run_iteration(iteration, current_text)

    def _run_iteration(self, data, current_text):
        """One autoregressive step: process → distribution → select → append."""
        llm = self.bb_box

        self.play(
            llm.box.animate.set_stroke(YELLOW, width=5),
            run_time=0.25,
        )
        self.play(
            llm.box.animate.set_stroke(ACCENT_COLOR, width=3),
            run_time=0.25,
        )

        arrow_out = Arrow(
            llm.get_right(), llm.get_right() + RIGHT * 0.9,
            buff=0.15, color=GRAY_B, stroke_width=2,
        )
        self.play(GrowArrow(arrow_out), run_time=0.4)

        dist = CompactDistribution(data["top_tokens"], data["probs"])
        dist.next_to(arrow_out, RIGHT, buff=0.25)
        self.play(
            LaggedStart(
                *[FadeIn(r, shift=LEFT * 0.2) for r in dist.token_groups],
                lag_ratio=0.08,
            ),
            run_time=0.8,
        )

        cond_eq = EquationLabel(
            r"p(x_t \mid x_{<t}; \theta) = \mathrm{softmax}(f_\theta(x_{<t}))",
            font_size=20,
        )
        cond_eq.next_to(dist, UP, buff=0.2).shift(RIGHT * 0.25)
        self.play(FadeIn(cond_eq), run_time=0.4)

        sel_idx = data["top_tokens"].index(data["selected"])
        highlight = SurroundingRectangle(
            dist.token_groups[sel_idx],
            color=YELLOW, stroke_width=2, buff=0.06,
        )
        self.play(Create(highlight), run_time=0.4)
        self.wait(0.4)

        sample_eq = EquationLabel(
            r"x_t \sim \mathrm{Categorical}\big(p(\cdot \mid x_{<t}; \theta)\big)",
            font_size=20,
        )
        sample_eq.move_to(cond_eq.get_center())
        self.play(Transform(cond_eq, sample_eq), run_time=0.4)

        selected = data["selected"]
        flying = Text(selected, font_size=22, color=YELLOW, weight=BOLD)
        flying.move_to(dist.token_groups[sel_idx][0])
        clamp_to_frame(flying)

        target = self.bb_sentence.get_right() + RIGHT * 0.25 + UP * 0.25
        self.play(FadeIn(flying, scale=1.2), run_time=0.2)
        self.play(
            flying.animate(path_arc=-PI / 3).move_to(target),
            run_time=0.9,
            rate_func=smooth,
        )

        if selected in {".", ",", "!", "?", ";", ":"}:
            new_text = current_text + selected
        else:
            new_text = current_text + " " + selected

        new_sentence = Text(f'"{new_text}"', font_size=20, color=TEXT_COLOR)
        new_sentence.move_to(self.bb_sentence.get_center())
        clamp_to_frame(new_sentence)

        new_arrow_in = Arrow(
            new_sentence.get_right(), self.bb_box.get_left(),
            buff=0.2, color=GRAY_B, stroke_width=2,
        )

        self.play(
            FadeOut(flying),
            FadeOut(dist), FadeOut(highlight), FadeOut(arrow_out), FadeOut(cond_eq),
            Transform(self.bb_sentence, new_sentence),
            Transform(self.bb_arrow_in, new_arrow_in),
            run_time=0.7,
        )
        self.wait(0.3)
        return new_text

    # ------------------------------------------------------------------
    # Reusable zoom helpers
    # ------------------------------------------------------------------

    def add_behind(self, *mobjects):
        for m in mobjects:
            self.add(m)
            self.bring_to_back(m)

    def zoom_into_block(self, target, box_attr, fade_out, question_text,
                        inner_fade_attrs, stroke_color=ACCENT_COLOR):
        """Reusable zoom-into transition. Returns zoom_cover for reveal."""
        question = Text(question_text, font_size=28, color=ACCENT_COLOR)
        # Center question on screen to avoid going off edges
        question.move_to(ORIGIN + DOWN * 2.5)
        clamp_to_frame(question)
        self.play(Write(question), run_time=1.2)
        self.wait(0.8)

        self.play(
            *[FadeOut(m) for m in fade_out],
            FadeOut(question),
            run_time=0.8,
        )

        self.play(target.animate.move_to(ORIGIN), run_time=0.4)

        box = getattr(target, box_attr)
        self.play(
            box.animate.set_stroke(YELLOW, width=6),
            rate_func=there_and_back,
            run_time=0.8,
        )

        anims = []
        for attr in inner_fade_attrs:
            anims.append(getattr(target, attr).animate.set_opacity(0))
        self.play(*anims, run_time=0.5)

        zoom_cover = box.copy()
        zoom_cover.set_fill(DARK_BG, opacity=1)
        zoom_cover.set_stroke(stroke_color, width=3)
        self.add(zoom_cover)
        self.remove(target)

        self.play(
            zoom_cover.animate.scale(6).set_stroke(width=0),
            run_time=2.0,
            rate_func=rush_into,
        )
        return zoom_cover

    def reveal_after_zoom(self, zoom_cover, *content):
        """Place content behind zoom_cover, then fade out cover."""
        self.add_behind(*content)
        self.play(
            FadeOut(zoom_cover),
            run_time=1.5,
            rate_func=rush_from,
        )

    # ------------------------------------------------------------------
    # PHASE 2 — Three-stage pipeline overview
    # ------------------------------------------------------------------

    def transition_to_three_stage(self):
        """Zoom into the LLM black box, reveal the three-stage pipeline title."""
        zoom_cover = self.zoom_into_block(
            target=self.bb_box,
            box_attr='box',
            fade_out=[self.bb_title, self.bb_sentence, self.bb_arrow_in, self.bb_eq],
            question_text="But how does this work?",
            inner_fade_attrs=['title_text', 'theta_text'],
        )

        title = Text(
            "Three Stages of a Language Model",
            font_size=28, color=ACCENT_COLOR,
        )
        title.to_edge(UP, buff=0.3)
        clamp_to_frame(title)
        self.reveal_after_zoom(zoom_cover, title)
        self.tsp_title = title

    def show_three_stage_pipeline(self):
        """Display the three-stage pipeline blocks and run autoregressive iterations."""
        input_block = PipelineStageBlock(
            title="Input Encoding",
            sub_items=["Tokenization", "Embedding Lookup", "Positional Encoding"],
            color=ACCENT_COLOR,
        )
        transformer_block = PipelineStageBlock(
            title="Transformer Core",
            sub_items=["Self-Attention", "Feed-Forward (FFN)",
                       "Residual Connections", "Layer Normalization"],
            color=TRANSFORMER_RED,
        )
        output_block = PipelineStageBlock(
            title="Output Decoding",
            sub_items=["Linear Projection", "Softmax", "Sampling"],
            color=ACCENT_COLOR_2,
        )

        input_block.move_to(LEFT * 4.4 + DOWN * 0.3)
        transformer_block.move_to(DOWN * 0.3)
        output_block.move_to(RIGHT * 4.4 + DOWN * 0.3)
        clamp_to_frame(input_block)
        clamp_to_frame(output_block)

        arrow_1 = Arrow(
            input_block.get_right(), transformer_block.get_left(),
            buff=0.15, color=GRAY_B, stroke_width=2,
        )
        arrow_2 = Arrow(
            transformer_block.get_right(), output_block.get_left(),
            buff=0.15, color=GRAY_B, stroke_width=2,
        )

        self.play(
            LaggedStart(
                FadeIn(input_block, shift=UP * 0.3),
                FadeIn(transformer_block, shift=UP * 0.3),
                FadeIn(output_block, shift=UP * 0.3),
                lag_ratio=0.20,
            ),
            run_time=1.5,
        )
        self.play(GrowArrow(arrow_1), GrowArrow(arrow_2), run_time=0.6)
        self.wait(0.5)

        self.tsp_input_block = input_block
        self.tsp_transformer_block = transformer_block
        self.tsp_output_block = output_block
        self.tsp_arrow_1 = arrow_1
        self.tsp_arrow_2 = arrow_2
        self.wait(1.0)

    # ------------------------------------------------------------------
    # Reusable zoom-back helper
    # ------------------------------------------------------------------

    def zoom_back_to_three_stage(self, handoff_from=None, handoff_to=None,
                                  matrix_label=None, narrative_text=None):
        """Fade out current content and restore the three-stage pipeline view.

        If handoff_from and handoff_to are specified (e.g., 'input', 'transformer', 'output'),
        show a matrix traveling between those stages with narrative text.
        """
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)

        title = Text(
            "Three Stages of a Language Model",
            font_size=28, color=ACCENT_COLOR,
        )
        title.to_edge(UP, buff=0.3)
        clamp_to_frame(title)

        input_block = PipelineStageBlock(
            title="Input Encoding",
            sub_items=["Tokenization", "Embedding Lookup", "Positional Encoding"],
            color=ACCENT_COLOR,
        )
        transformer_block = PipelineStageBlock(
            title="Transformer Core",
            sub_items=["Self-Attention", "Feed-Forward (FFN)",
                       "Residual Connections", "Layer Normalization"],
            color=TRANSFORMER_RED,
        )
        output_block = PipelineStageBlock(
            title="Output Decoding",
            sub_items=["Linear Projection", "Softmax", "Sampling"],
            color=ACCENT_COLOR_2,
        )

        input_block.move_to(LEFT * 4.4 + DOWN * 0.3)
        transformer_block.move_to(DOWN * 0.3)
        output_block.move_to(RIGHT * 4.4 + DOWN * 0.3)
        clamp_to_frame(input_block)
        clamp_to_frame(output_block)

        arrow_1 = Arrow(
            input_block.get_right(), transformer_block.get_left(),
            buff=0.15, color=GRAY_B, stroke_width=2,
        )
        arrow_2 = Arrow(
            transformer_block.get_right(), output_block.get_left(),
            buff=0.15, color=GRAY_B, stroke_width=2,
        )

        self.play(
            FadeIn(title),
            FadeIn(input_block), FadeIn(transformer_block), FadeIn(output_block),
            FadeIn(arrow_1), FadeIn(arrow_2),
            run_time=1.0,
        )

        self.tsp_title = title
        self.tsp_input_block = input_block
        self.tsp_transformer_block = transformer_block
        self.tsp_output_block = output_block
        self.tsp_arrow_1 = arrow_1
        self.tsp_arrow_2 = arrow_2

        # ── Handoff animation: matrix emerges from block, drops, then travels ──
        if handoff_from and handoff_to:
            block_map = {
                'input': input_block,
                'transformer': transformer_block,
                'output': output_block,
            }
            color_map = {
                'input': ACCENT_COLOR,
                'transformer': TRANSFORMER_RED,
                'output': ACCENT_COLOR_2,
            }

            from_block = block_map[handoff_from]
            to_block = block_map[handoff_to]

            # Show narrative question FIRST
            if narrative_text:
                narr = Text(narrative_text, font_size=20, color=GRAY_A)
                narr.set(width=wrap_width)
                narr.to_edge(DOWN, buff=0.6)
                clamp_to_frame(narr)
                self.play(FadeIn(narr, shift=UP * 0.2), run_time=0.6)
                self.wait(1.2)
                self.play(FadeOut(narr), run_time=0.4)

            # ── Create traveling matrix + label ──
            mat = TensorMatrix(3, 4, cell_size=0.12, seed=99)

            label_text = matrix_label or "X"
            mat_label = MathTex(label_text, font_size=16, color=color_map[handoff_from])
            mat_label.next_to(mat, DOWN, buff=0.08)

            mat_group = VGroup(mat, mat_label)
            mat_group.scale(1.2)

            spawn_pos = from_block.get_bottom()
            rest_pos = from_block.get_bottom() + DOWN * 0.6
            target_pos = to_block.get_bottom() + DOWN * 0.6

            mat_group.move_to(spawn_pos)
            clamp_to_frame(mat_group)

            # Stage 1 — Emerge from block
            self.play(GrowFromEdge(mat_group, UP), run_time=0.7)

            # Stage 2 — Drop to resting position
            self.play(
                mat_group.animate.move_to(rest_pos),
                run_time=0.6,
                rate_func=smooth,
            )

            self.wait(0.6)

            # Determine arrow
            if handoff_from == 'input' and handoff_to == 'transformer':
                travel_arrow = arrow_1
            elif handoff_from == 'transformer' and handoff_to == 'output':
                travel_arrow = arrow_2
            else:
                travel_arrow = None

            if travel_arrow:
                self.play(travel_arrow.animate.set_color(YELLOW), run_time=0.3)

            # Stage 3 — Travel to next block
            self.play(
                mat_group.animate.move_to(target_pos).scale(1.0),
                run_time=1.2,
                rate_func=smooth,
            )

            # Change label color AFTER movement (prevents desync)
            self.play(
                mat_group[1].animate.set_color(color_map[handoff_to]),
                run_time=0.3,
            )


            if travel_arrow:
                self.play(travel_arrow.animate.set_color(GRAY_B), run_time=0.3)

            self.wait(0.6)

            # Fade out before zooming in
            self.play(FadeOut(mat_group), run_time=0.5)


    # ------------------------------------------------------------------
    # PHASE 2.5 — Zoom into each stage
    # ------------------------------------------------------------------

    def transition_to_input_encoding(self):
        """Zoom into the Input Encoding block with narrative motivation."""
        # ── Narrative: Why do we need Input Encoding? ──
        motivation = Text(
            "Transformers can't process raw text—they need numbers.",
            font_size=22, color=GRAY_A,
        )
        motivation.to_edge(DOWN, buff=0.8)
        clamp_to_frame(motivation)
        self.play(FadeIn(motivation, shift=UP * 0.2), run_time=0.8)
        self.wait(1.0)
        self.play(FadeOut(motivation), run_time=0.5)

        zoom_cover = self.zoom_into_block(
            target=self.tsp_input_block,
            box_attr='box',
            fade_out=[self.tsp_title, self.tsp_transformer_block,
                      self.tsp_output_block, self.tsp_arrow_1, self.tsp_arrow_2],
            question_text="How do we convert text to numbers?",
            inner_fade_attrs=['title_text', 'separator', 'sub_labels'],
        )

        inner_title = Text(
            "Input Encoding",
            font_size=32, color=ACCENT_COLOR,
        )
        inner_title.to_edge(UP, buff=0.5)
        clamp_to_frame(inner_title)

        sentence_label = Text("Input:", font_size=20, color=GRAY_B)
        sentence_text = Text(f'"{SENTENCE}"', font_size=28, color=TEXT_COLOR)
        sentence_group = VGroup(sentence_label, sentence_text).arrange(RIGHT, buff=0.3)
        sentence_group.next_to(inner_title, DOWN, buff=0.8)
        clamp_to_frame(sentence_group)

        self.reveal_after_zoom(zoom_cover, inner_title, sentence_label, sentence_text)

        self.title = inner_title
        self.sentence_text = sentence_text
        self.sentence_label = sentence_label

    def transition_to_transformer_core(self):
        """Zoom into the Transformer Core block with narrative motivation."""
        # ── Narrative: Why do we need the Transformer? ──
        # motivation = Text(
        #     "We have numbers... but they're just random embeddings.",
        #     font_size=22, color=GRAY_A,
        # )
        # motivation.to_edge(DOWN, buff=0.8)
        # self.play(FadeIn(motivation, shift=UP * 0.2), run_time=0.8)
        # self.wait(1.0)

        # motivation2 = Text(
        #     "How do we make sense of this and predict the next word?",
        #     font_size=22, color=GRAY_A,
        # )
        # motivation2.to_edge(DOWN, buff=0.8)
        # self.play(Transform(motivation, motivation2), run_time=0.6)
        # self.wait(1.0)
        # self.play(FadeOut(motivation), run_time=0.5)

        zoom_cover = self.zoom_into_block(
            target=self.tsp_transformer_block,
            box_attr='box',
            fade_out=[self.tsp_title, self.tsp_input_block,
                      self.tsp_output_block, self.tsp_arrow_1, self.tsp_arrow_2],
            question_text="This is where the magic happens...",
            inner_fade_attrs=['title_text', 'separator', 'sub_labels'],
            stroke_color=TRANSFORMER_RED,
        )

        # Just fade out the zoom cover - show_transformer_pipeline will handle content
        self.play(FadeOut(zoom_cover), run_time=0.8)

    def transition_to_output_decoding(self):
        """Zoom into the Output Decoding block with narrative motivation."""
        # ── Narrative: Why do we need Output Decoding? ──
        motivation = Text(
            "Nice — we now have powerful, context-aware embeddings.",
            font_size=24, color=YELLOW,
        )
        motivation.to_edge(DOWN, buff=0.9)
        clamp_to_frame(motivation)
        self.play(FadeIn(motivation, shift=UP * 0.2), run_time=0.8)
        self.wait(1.0)

        motivation2 = Text(
            "But embeddings aren't words… you can't read them.",
            font_size=24, color=YELLOW,
        )
        motivation2.to_edge(DOWN, buff=0.9)
        clamp_to_frame(motivation2)
        self.play(Transform(motivation, motivation2), run_time=0.6)
        self.wait(1.0)
        
        # motivation3 = Text(
        #     "So how do we turn this into an actual next token?",
        #     font_size=22, color=GRAY_A,
        # )
        # motivation3.to_edge(DOWN, buff=0.9)
        # self.play(Transform(motivation2, motivation3), run_time=0.6)
        # self.wait(1.0)
        self.play(FadeOut(motivation), run_time=0.5)

        zoom_cover = self.zoom_into_block(
            target=self.tsp_output_block,
            box_attr='box',
            fade_out=[self.tsp_title, self.tsp_input_block,
                      self.tsp_transformer_block, self.tsp_arrow_1, self.tsp_arrow_2],
            question_text="So how do we turn this into an actual next word?",
            inner_fade_attrs=['title_text', 'separator', 'sub_labels'],
            stroke_color=ACCENT_COLOR_2,
        )

        # Just fade out the zoom cover - show_output_decoding will handle content
        self.play(FadeOut(zoom_cover), run_time=0.8)

    # ------------------------------------------------------------------
    # PHASE 3 — Detailed pipeline
    # ------------------------------------------------------------------

    def show_tokenization(self):
        """Text → token boxes."""
        stage_label = Text("1. Tokenization", font_size=24, color=ACCENT_COLOR_2)
        stage_label.to_edge(LEFT, buff=0.5).shift(UP * 2)
        clamp_to_frame(stage_label)

        token_row = TokenRow(TOKENS)
        token_row.next_to(self.sentence_text, DOWN, buff=1)

        arrow = Arrow(
            self.sentence_text.get_bottom(), token_row.get_top(),
            buff=0.2, color=GRAY_B, stroke_width=2,
        )

        eq = EquationLabel(
            [
                r"x_{1:T} = \text{tokenize}(\text{text})",
                r"[x_1,\ldots,x_T] = [\text{tok}_1,\ldots,\text{tok}_T]",
            ],
            font_size=28,
        )
        eq.scale(0.7).to_edge(RIGHT, buff=0.8).shift(UP * 1.5)
        clamp_to_frame(eq)

        self.play(FadeIn(stage_label))
        self.play(GrowArrow(arrow), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(b, scale=0.8) for b in token_row], lag_ratio=0.1),
            run_time=1.5,
        )
        self.play(FadeIn(eq), run_time=0.8)
        self.wait(0.5)

        self.token_row = token_row
        self.token_arrow = arrow
        self.stage_label = stage_label
        self.current_eq = eq

    def show_token_ids(self):
        """Tokens → integer IDs."""
        new_stage = Text("2. Token IDs", font_size=24, color=ACCENT_COLOR_2)
        new_stage.move_to(self.stage_label)
        clamp_to_frame(new_stage)

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

        new_eq = EquationLabel(
            [
                r"x_t = \text{id}(\text{tok}_t)",
                r"x_t \in \{1,\dots,|V|\}",
            ],
            font_size=26,
        )
        new_eq.scale(0.7).move_to(self.current_eq)
        clamp_to_frame(new_eq)

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
        """IDs → embedding-matrix lookup: show W_E, extract rows, build E."""
        # ── Rearrange previous elements ──
        self.play(
            FadeOut(self.title), FadeOut(self.sentence_text),
            FadeOut(self.sentence_label), FadeOut(self.token_arrow),
            self.token_row.animate.scale(0.65).to_edge(UP, buff=0.35),
            self.id_row.animate.scale(0.65).to_edge(UP, buff=1.05),
            FadeOut(self.id_arrows),
            run_time=1,
        )

        new_stage = Text("3. Embedding Lookup", font_size=24, color=ACCENT_COLOR_2)
        new_stage.move_to(self.stage_label).shift(RIGHT * 0.5)
        clamp_to_frame(new_stage)

        new_eq = EquationLabel(
            [
                r"e_t = W_E[x_t]",
                r"E = [e_1;\dots;e_T] \in \mathbb{R}^{t \times d}",
            ],
            font_size=26,
        )
        new_eq.scale(0.7).move_to(self.current_eq)
        clamp_to_frame(new_eq)

        # ── Weight matrix W_E: |V| × d ──
        weight_mat = EmbeddingWeightMatrix(num_cols=D_MODEL, cell_size=0.22)
        weight_mat.move_to(LEFT * 2.5 + DOWN * 0.5)

        w_label = MathTex(r"W_E", font_size=26, color=ACCENT_COLOR)
        w_label.next_to(weight_mat, DOWN, buff=0.2)
        clamp_to_frame(w_label)

        brace_v = Brace(weight_mat, LEFT, buff=0.1, color=GRAY_B)
        v_label = MathTex(r"|V|", font_size=18, color=GRAY_B)
        v_label.next_to(brace_v, LEFT, buff=0.08)
        clamp_to_frame(v_label)

        brace_d_w = Brace(weight_mat, UP, buff=0.1, color=GRAY_B)
        d_label_w = MathTex(r"d", font_size=18, color=GRAY_B)
        d_label_w.next_to(brace_d_w, UP, buff=0.05)
        clamp_to_frame(d_label_w)

        weight_group = VGroup(
            weight_mat, w_label, brace_v, v_label, brace_d_w, d_label_w,
        )

        self.play(Transform(self.stage_label, new_stage))
        self.play(
            FadeIn(weight_mat, scale=0.9),
            FadeIn(w_label),
            FadeIn(brace_v), FadeIn(v_label),
            FadeIn(brace_d_w), FadeIn(d_label_w),
            Transform(self.current_eq, new_eq),
            run_time=1.5,
        )
        self.wait(0.5)

        # ── Row extraction: each token-ID → one row from W_E ──
        extracted_rows = VGroup()
        result_center = RIGHT * 2.5 + DOWN * 0.5
        first_row_y = result_center[1] + (T - 1) * weight_mat.cell_size / 2

        for i in range(T):
            row = weight_mat.make_extracted_row(TOKEN_IDS[i])
            row.move_to(weight_mat.get_center())

            target_y = first_row_y - i * weight_mat.cell_size
            target_pos = np.array([result_center[0], target_y, 0])

            # Highlight the current token ID
            id_hl = SurroundingRectangle(
                self.id_row[i], buff=0.04, color=YELLOW, stroke_width=2,
            )
            # Arrow from token ID to weight matrix
            top_center = weight_mat.border.get_top()
            offset = (i - (T - 1) / 2) * 0.15
            arrow = Arrow(
                self.id_row[i].get_bottom(),
                top_center + RIGHT * offset,
                buff=0.12, color=YELLOW, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.12,
            )

            self.play(Create(id_hl), GrowArrow(arrow), run_time=0.25)
            self.play(FadeIn(row, scale=0.6), run_time=0.15)
            self.play(
                row.animate.move_to(target_pos),
                FadeOut(id_hl), FadeOut(arrow),
                run_time=0.4,
            )
            extracted_rows.add(row)

        # ── Border and labels around the stacked result: E (t × d) ──
        result_border = SurroundingRectangle(
            extracted_rows, buff=0.04,
            stroke_color=WHITE, stroke_width=2, fill_opacity=0,
        )
        e_label = MathTex(r"E", font_size=24, color=ACCENT_COLOR)
        e_label.next_to(result_border, DOWN, buff=0.15)
        clamp_to_frame(e_label)

        brace_t = Brace(result_border, LEFT, buff=0.1, color=GRAY_B)
        t_label = MathTex(r"t", font_size=18, color=GRAY_B)
        t_label.next_to(brace_t, LEFT, buff=0.05)
        clamp_to_frame(t_label)

        brace_d_e = Brace(result_border, UP, buff=0.1, color=GRAY_B)
        d_label_e = MathTex(r"d", font_size=18, color=GRAY_B)
        d_label_e.next_to(brace_d_e, UP, buff=0.05)
        clamp_to_frame(d_label_e)

        result_decor = VGroup(
            result_border, e_label, brace_t, t_label, brace_d_e, d_label_e,
        )

        self.play(
            Create(result_border), FadeIn(e_label),
            FadeIn(brace_t), FadeIn(t_label),
            FadeIn(brace_d_e), FadeIn(d_label_e),
            run_time=0.8,
        )
        self.wait(0.5)

        # ── Transition: replace with a clean TensorMatrix for next phase ──
        emb_matrix = TensorMatrix(T, D_MODEL, cell_size=0.22, label="E: t × d")
        emb_matrix.move_to(DOWN * 0.5)

        self.play(
            FadeOut(weight_group),
            FadeOut(result_decor),
            FadeOut(extracted_rows),
            run_time=1.2,
        )
        self.play(FadeIn(emb_matrix), run_time=0.6)
        self.wait(0.3)

        self.emb_matrix = emb_matrix

    def show_positional_encoding(self):
        """Positional encoding: X₀ = E + P."""
        # ── Clean up previous elements for space ──
        self.play(
            FadeOut(self.token_row), FadeOut(self.id_row),
            run_time=0.6,
        )

        # ── Update stage label ──
        new_stage = Text("4. Positional Encoding", font_size=24, color=ACCENT_COLOR_2)
        new_stage.move_to(self.stage_label)
        clamp_to_frame(new_stage)
        self.play(Transform(self.stage_label, new_stage), run_time=0.5)

        # ── Reposition E matrix to the left ──
        e_mat = self.emb_matrix
        self.play(e_mat.animate.move_to(LEFT * 4.2 + DOWN * 0.3), run_time=0.5)

        # ── Build E + P = X₀ layout ──
        plus = MathTex("+", font_size=32, color=TEXT_COLOR)
        plus.next_to(e_mat, RIGHT, buff=0.3)
        clamp_to_frame(plus)

        pe_mat = TensorMatrix(T, D_MODEL, cell_size=0.22, label="P: t × d", seed=77)
        pe_mat.next_to(plus, RIGHT, buff=0.3)

        equals = MathTex("=", font_size=32, color=TEXT_COLOR)
        equals.next_to(pe_mat, RIGHT, buff=0.3)
        clamp_to_frame(equals)

        x0_mat = TensorMatrix(T, D_MODEL, cell_size=0.22, label="X₀: t × d", seed=99)
        x0_mat.next_to(equals, RIGHT, buff=0.3)

        # ── Equation ──
        new_eq = EquationLabel(
            [
                r"X_0 = E + P",
                r"PE_{(pos,2i)} = \sin\!\left(\tfrac{pos}{10000^{2i/d}}\right)",
                r"PE_{(pos,2i+1)} = \cos\!\left(\tfrac{pos}{10000^{2i/d}}\right)",
            ],
            font_size=22,
        )
        new_eq.scale(0.7).to_edge(RIGHT, buff=0.3).shift(UP * 2.5)
        clamp_to_frame(new_eq)

        # ── Animate ──
        self.play(FadeIn(plus), FadeIn(pe_mat, shift=DOWN * 0.2), run_time=0.8)
        self.play(Transform(self.current_eq, new_eq), run_time=0.8)
        self.wait(0.5)

        # ── Show the summation result ──
        self.play(FadeIn(equals), FadeIn(x0_mat, shift=DOWN * 0.2), run_time=0.8)
        self.wait(0.5)

        # ── Brief highlight: sine-wave intuition ──
        intuition = Text(
            "Each token gets a unique signature from sin/cos waves", #Here we will mention that this is from the original transformers paper, now new techniques are widely used like learnable pos embeddings, and RoPE; we will discuss those later.
            font_size=16, color=GRAY_B,
        )

        intuition.next_to(VGroup(e_mat, x0_mat), DOWN, buff=0.6)
        clamp_to_frame(intuition)
        self.play(FadeIn(intuition), run_time=0.5)
        self.wait(1.0)
        self.play(FadeOut(intuition), run_time=0.4)

        # ── Narrative conclusion: X₀ is ready ──
        conclusion = Text(
            "X₀: Position-aware embeddings, ready for the Transformer!",
            font_size=20, color=ACCENT_COLOR,
        )
        conclusion.next_to(VGroup(e_mat, x0_mat), DOWN, buff=0.6)
        clamp_to_frame(conclusion)
        self.play(FadeIn(conclusion, shift=UP * 0.2), run_time=0.6)
        self.wait(1.0)

        # ── Store X₀ for handoff animation ──
        self.x0_matrix = x0_mat
        self.x0_conclusion = conclusion

    # ------------------------------------------------------------------
    # PHASE 4 — Transformer core pipeline
    # ------------------------------------------------------------------

    def show_transformer_pipeline(self):
        """Transformer core: MHA → Add&Norm → FFN → Add&Norm with equations."""
        # ── Clear previous phase ──
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=2.0)

        title = Text("Transformer Core", font_size=28, color=TRANSFORMER_RED)
        title.to_edge(UP, buff=0.25)
        clamp_to_frame(title)

        # ── Layout constants ──
        BW, BH = 2.5, 0.38
        GAP = 0.34
        PIPELINE_X = -1.5
        MAT_X = PIPELINE_X - BW / 2 - 1.2
        EQ_X = 3.5

        # ── Block definitions (core transformer only) ──
        block_data = [
            ("Masked Multi-Head Self-Attention \n\t\t\t\t\t(MHA)",  TRANSFORMER_RED, 10),
            ("Add & Norm",                   "#ffb74d",   13),
            ("Feed Forward Network \n\t\t\t\t(FFN)",  TRANSFORMER_RED, 13),
            ("Add & Norm",                   "#ffb74d",   13),
        ]
        orig_colors = [c for _, c, _ in block_data]

        def make_block(label, color, fs):
            box = RoundedRectangle(
                width=BW, height=BH, corner_radius=0.08,
                stroke_color=color, stroke_width=2,
                fill_color=BOX_COLOR, fill_opacity=0.75,
            )
            txt = Text(label, font_size=fs, color=color)
            txt.move_to(box)
            return VGroup(box, txt)

        blocks = [make_block(*d) for d in block_data]

        pipeline = VGroup(*blocks)
        pipeline.arrange(UP, buff=GAP)
        pipeline.move_to(np.array([PIPELINE_X, -0.15, 0]))

        # ── Connecting arrows ──
        arrows = VGroup()
        for i in range(len(blocks) - 1):
            a = Arrow(
                blocks[i].get_top(), blocks[i + 1].get_bottom(),
                buff=0.02, color=GRAY_B, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.25,
            )
            arrows.add(a)

        # ── Container with ×N label ──
        container = SurroundingRectangle(
            pipeline, buff=0.25,
            stroke_color=GRAY_A, stroke_width=1.5, fill_opacity=0.05,
        )
        brace = Brace(container, RIGHT, buff=0.15, color=GRAY_B)
        n_label = MathTex(r"\times N", font_size=18, color=GRAY_B)
        n_label.next_to(brace, RIGHT, buff=0.08)
        clamp_to_frame(n_label)

        # ── Residual arrows ──
        from manim import CubicBezier

        res_color = "#66bb6a"

        def residual_ortho(
            flow_arrow, dst_block,
            x_offset=1.6, bend_radius=0.18, into_box=0.06, stroke_w=2.2,
        ):
            start = flow_arrow.get_center() + RIGHT * 0.05
            end_y = dst_block.get_center()[1]
            end_x = dst_block[0].get_right()[0] - into_box
            end = np.array([end_x, end_y, 0])
            bend_x = start[0] + x_offset

            A = start
            B = np.array([bend_x, start[1], 0])
            C = np.array([bend_x, end_y, 0])
            D = end

            h, v, l = RIGHT, (UP if C[1] > B[1] else DOWN), LEFT
            r = bend_radius

            A2 = B - h * r
            B2 = B + v * r
            C1 = C - v * r
            C2 = C + l * r

            path = VGroup(
                Line(A, A2),
                CubicBezier(A2, A2 + h * r, B2 - v * r, B2),
                Line(B2, C1),
                CubicBezier(C1, C1 + v * r, C2 - l * r, C2),
                Line(C2, D),
            )
            path.set_stroke(res_color, width=stroke_w).set_z_index(3)

            tip = ArrowTriangleFilledTip(color=res_color)
            tip.scale(0.35)
            tip.rotate(angle_of_vector(D - C2) + PI)
            tip.shift(D - tip.get_tip_point())
            tip.set_z_index(4)
            return VGroup(path, tip)

        # Input arrow (X₀ entering from below)
        input_arrow = Arrow(
            blocks[0].get_bottom() + DOWN * 0.6, blocks[0].get_bottom(),
            buff=0.02, color=GRAY_B, stroke_width=1.5,
            max_tip_length_to_length_ratio=0.25,
        )
        input_label = MathTex("X_0", font_size=18, color=ACCENT_COLOR)
        input_label.next_to(input_arrow, DOWN, buff=0.05)
        clamp_to_frame(input_label)

        # Output arrow (X₂ leaving from top)
        output_arrow = Arrow(
            blocks[-1].get_top(), blocks[-1].get_top() + UP * 0.6,
            buff=0.02, color=GRAY_B, stroke_width=1.5,
            max_tip_length_to_length_ratio=0.25,
        )
        output_label = MathTex("X_N", font_size=18, color=ACCENT_COLOR)
        output_label.next_to(output_arrow, UP, buff=0.05)
        clamp_to_frame(output_label)

        # Attention residual (skip around MHA → Add&Norm)
        res1 = residual_ortho(arrows[0], blocks[1], x_offset=1.37)
        # FFN residual (skip around FFN → Add&Norm)
        res2 = residual_ortho(arrows[1], blocks[3], x_offset=1.37)

        # ── Phase 1: Show pipeline structure ──
        self.play(FadeIn(title), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.05) for b in blocks], lag_ratio=0.05),
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05),
            run_time=1.2,
        )
        self.play(
            Create(container),
            GrowFromCenter(brace), FadeIn(n_label),
            GrowArrow(input_arrow), FadeIn(input_label),
            GrowArrow(output_arrow), FadeIn(output_label),
            Create(res1), Create(res2),
            run_time=0.6,
        )
        self.wait(0.5)

        # ── Phase 2: Traveling matrix with per-stage equations ──
        mat = TensorMatrix(T, D_MODEL, cell_size=0.15, label=None, seed=99)
        mat.move_to(np.array([MAT_X, blocks[0].get_center()[1] - 0.8, 0]))

        mat_label = MathTex("X_0", font_size=20, color=ACCENT_COLOR)
        mat_label.next_to(mat, DOWN, buff=0.1)
        clamp_to_frame(mat_label)

        self.play(FadeIn(mat, shift=UP * 0.3), FadeIn(mat_label), run_time=0.5)

        # Stage definitions: (active_block_indices, equation_lines, new_label, font_size)
        stages = [
            (
                [0],
                [
                    r"Q = XW_Q,\; K = XW_K,\; V = XW_V",
                    r"M_{ij}=\begin{cases}0,& j\le i\\ -\infty,& j>i\end{cases}",
                    r"\text{head}_i=\text{softmax}\!\left(\tfrac{Q_iK_i^\top}{\sqrt{d_k}}+M\right)V_i",
                    r"\text{MHA}(X)=[\text{head}_1 \cdots \text{head}_h]\,W_O",
                ],
                None,
                20,
            ),
            (
                [1],
                [
                    r"X_1 = \text{LayerNorm}\!\left(X_0 + \text{MHA}(X_0)\right)",
                    r"\text{LN}(X)=\frac{X-\mu}{\sqrt{\sigma^2+\epsilon}}\odot\gamma+\beta",
                ],
                "X_1",
                20,
            ),
            (
                [2],
                [
                    r"\text{FFN}(X) = \sigma(XW_1 + b_1)\,W_2 + b_2",
                    r"\sigma=\text{GELU}",
                ],
                None,
                20,
            ),
            (
                [3],
                [r"X_2 = \text{LayerNorm}\!\left(X_1 + \text{FFN}(X_1)\right)"],
                "X_2",
                20,
            ),
        ]

        prev_eq = None

        for s_blocks, eq_lines, new_label, eq_font_size in stages:
            target_y = np.mean([blocks[i].get_center()[1] for i in s_blocks])
            target_pos = np.array([MAT_X, target_y, 0])
            label_pos = np.array([MAT_X, target_y - mat.height / 2 - 0.12, 0])

            eq_panel = EquationLabel(eq_lines, font_size=eq_font_size)
            eq_panel.move_to(np.array([EQ_X, target_y, 0]))
            clamp_to_frame(eq_panel)

            if eq_panel.get_right()[0] > 6.8:
                eq_panel.shift(LEFT * (eq_panel.get_right()[0] - 6.8))

            anims = [mat.animate.move_to(target_pos)]

            if new_label:
                new_lbl = MathTex(new_label, font_size=20, color=ACCENT_COLOR)
                new_lbl.move_to(label_pos)
                clamp_to_frame(new_lbl)
                anims.append(Transform(mat_label, new_lbl))
            else:
                anims.append(mat_label.animate.move_to(label_pos))

            for idx in s_blocks:
                anims.append(blocks[idx][0].animate.set_stroke(YELLOW, width=3))

            self.play(*anims, run_time=0.6)

            if prev_eq:
                self.play(FadeOut(prev_eq, shift=UP * 0.15), run_time=0.25)

            self.play(FadeIn(eq_panel.background), run_time=0.3)
            self.play(
                LaggedStart(*[FadeIn(eq) for eq in eq_panel.equations], lag_ratio=0.3),
                run_time=max(0.7, len(eq_lines) * 0.45),
            )
            self.add(eq_panel)
            prev_eq = eq_panel
            self.wait(1.0)

            self.play(
                *[blocks[idx][0].animate.set_stroke(orig_colors[idx], width=2)
                  for idx in s_blocks],
                run_time=0.3,
            )

        if prev_eq:
            self.play(FadeOut(prev_eq), run_time=0.5)

        # ── Narrative conclusion: X_N is ready (step-by-step) ──
        wrap_width = config.frame_width * 0.85

        conclusion_q = Text(
            "Does the transformer core transform our embedding matrix into another embedding matrix?",
            font_size=20,
            color=TRANSFORMER_RED,
        )
        conclusion_q.set(width=wrap_width)
        conclusion_q.to_edge(DOWN, buff=0.8)
        clamp_to_frame(conclusion_q)

        conclusion_a = Text(
            "Yes — but now it contains rich, context-aware representations of each token based on the full sequence.",
            font_size=20,
            color=TRANSFORMER_RED,
        )
        conclusion_a.set(width=wrap_width)
        conclusion_a.move_to(conclusion_q)  # keeps transition visually clean
        clamp_to_frame(conclusion_a)

        self.play(FadeIn(conclusion_q, shift=UP * 0.2), run_time=0.6)
        self.wait(0.8)

        # Crossfade Q → A
        self.play(
            FadeOut(conclusion_q, shift=UP * 0.1),
            FadeIn(conclusion_a, shift=UP * 0.1),
            run_time=0.6,
        )

        self.wait(1.0)
        self.play(FadeOut(conclusion_a), run_time=0.4)

        self.wait(0.5)



    # ------------------------------------------------------------------
    # PHASE 5 — Output decoding pipeline
    # ------------------------------------------------------------------

    def show_output_decoding(self):
        """Linear projection → softmax → next-token probabilities.

        Shows that we only use the final row (x_T) which contains context
        from all previous tokens via attention.
        """
        # ── Clear previous phase ──
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)

        title = Text("Output Decoding", font_size=28, color=ACCENT_COLOR_2)
        title.to_edge(UP, buff=0.25)
        clamp_to_frame(title)
        self.play(FadeIn(title), run_time=0.5)

        subtitle = Text(
            "Converts embeddings into a probability distribution over the vocabulary, then selects the next token.",
            font_size=18,
            color=GRAY_A,
        )
        subtitle.next_to(title, DOWN, buff=0.15)
        clamp_to_frame(subtitle)
        self.play(FadeIn(subtitle), run_time=0.6)
        self.wait(0.3)

        # ── Phase 1: Show X_N and explain we only need the last row ──
        xn_mat = TensorMatrix(T, D_MODEL, cell_size=0.22, label="X_N", seed=55)
        xn_mat.move_to(LEFT * 4 + DOWN * 0.3)

        self.play(FadeIn(xn_mat, shift=UP * 0.3), run_time=0.6)

        # Narrative: explain why we only need the last row
        explain1 = Text(
            "X_N has T rows—one embedding per token position.",
            font_size=18, color=GRAY_A,
        )
        explain1.next_to(xn_mat, DOWN, buff=0.5)
        clamp_to_frame(explain1)
        self.play(FadeIn(explain1), run_time=0.5)
        self.wait(0.8)

        explain2 = Text(
            "But attention lets x_T 'see' all previous tokens!",
            font_size=18, color=ACCENT_COLOR,
        )
        explain2.next_to(explain1, DOWN, buff=0.2)
        clamp_to_frame(explain2)
        self.play(FadeIn(explain2), run_time=0.5)
        self.wait(1.0)

        # Highlight the last row
        last_row_cells = VGroup(*[xn_mat.get_cell(T - 1, j) for j in range(D_MODEL)])
        last_row_highlight = SurroundingRectangle(
            last_row_cells, color=YELLOW, stroke_width=2, buff=0.02,
        )
        self.play(Create(last_row_highlight), run_time=0.4)

        explain3 = Text(
            "We only need x_T to predict the next token.",
            font_size=18, color=YELLOW,
        )
        explain3.next_to(explain2, DOWN, buff=0.2)
        clamp_to_frame(explain3)
        self.play(FadeIn(explain3), run_time=0.5)
        self.wait(1.0)

        # Extract the last row as a vector
        xt_vec = TensorMatrix(1, D_MODEL, cell_size=0.22, label="x_T", seed=55)
        xt_vec.move_to(LEFT * 4 + DOWN * 0.3)

        self.play(
            FadeOut(explain1), FadeOut(explain2), FadeOut(explain3),
            FadeOut(last_row_highlight),
            run_time=0.5,
        )
        self.play(
            FadeOut(xn_mat),
            FadeIn(xt_vec),
            run_time=0.8,
        )
        self.wait(0.3)

        # ── Phase 2: Show vocabulary matrix multiplication ──
        # x_T (1 × d) · W_vocab (d × |V|) = z_T (1 × |V|)

        times_sign = MathTex(r"\times", font_size=28, color=TEXT_COLOR)
        times_sign.next_to(xt_vec, RIGHT, buff=1.0)
        clamp_to_frame(times_sign)

        # Create vocabulary matrix (d × |V|) - shown as d rows, few columns with ellipsis
        vocab_mat = EmbeddingWeightMatrix(num_cols=D_MODEL, cell_size=0.18, top_rows=3, bottom_rows=2)
        vocab_mat.next_to(times_sign, RIGHT, buff=1.0)

        vocab_label = Text("Vocabulary Matrix", font_size=14, color=ACCENT_COLOR_2)
        vocab_label.next_to(vocab_mat, UP, buff=1.30)
        clamp_to_frame(vocab_label)

        w_label = MathTex(r"W_{\text{vocab}}", font_size=20, color=ACCENT_COLOR_2)
        w_label.next_to(vocab_mat, DOWN, buff=0.15)
        clamp_to_frame(w_label)

        # Dimension labels
        brace_d = Brace(vocab_mat, LEFT, buff=0.08, color=GRAY_B)
        d_label = MathTex(r"d", font_size=14, color=GRAY_B)
        d_label.next_to(brace_d, LEFT, buff=0.05)
        clamp_to_frame(d_label)

        brace_v = Brace(vocab_mat, UP, buff=0.08, color=GRAY_B)
        v_label = MathTex(r"|V|", font_size=14, color=GRAY_B)
        v_label.next_to(brace_v, UP, buff=0.05)
        clamp_to_frame(v_label)

        self.play(
            FadeIn(times_sign),
            FadeIn(vocab_mat),
            FadeIn(vocab_label),
            FadeIn(w_label),
            run_time=0.8,
        )
        self.play(
            FadeIn(brace_d), FadeIn(d_label),
            FadeIn(brace_v), FadeIn(v_label),
            run_time=0.5,
        )

        # Equals sign and result
        equals_sign = MathTex(r"=", font_size=28, color=TEXT_COLOR)
        equals_sign.next_to(vocab_mat, RIGHT, buff=0.3)
        clamp_to_frame(equals_sign)

        # z_T is 1 × |V| (logits over vocabulary)
        zt_vec = TensorMatrix(1, D_MODEL, cell_size=0.18, label="z_T (logits)", seed=123)
        zt_vec.next_to(equals_sign, RIGHT, buff=0.3)

        self.play(FadeIn(equals_sign), FadeIn(zt_vec, shift=LEFT * 0.2), run_time=0.8)

        # Equation panel
        eq1 = EquationLabel(
            [r"z_T = x_T \cdot W_{\text{vocab}} + b",
             r"z_T \in \mathbb{R}^{|V|}"],
            font_size=20,
        )
        eq1.scale(0.8).to_edge(RIGHT, buff=0.3).shift(UP * 2.5)
        clamp_to_frame(eq1)
        self.play(FadeIn(eq1), run_time=0.6)
        self.wait(1.0)

        # ── Phase 3: Softmax ──
        softmax_arrow = Arrow(
            zt_vec.get_bottom() + DOWN * 0.1,
            zt_vec.get_bottom() + DOWN * 0.7,
            buff=0, color=ACCENT_COLOR_2, stroke_width=2,
        )
        softmax_label = Text("softmax", font_size=14, color=ACCENT_COLOR_2)
        softmax_label.next_to(softmax_arrow, RIGHT, buff=0.1)
        clamp_to_frame(softmax_label)

        pt_vec = TensorMatrix(1, D_MODEL, cell_size=0.18, label="p_T (probs)", seed=200)
        pt_vec.next_to(softmax_arrow, DOWN, buff=0.15)

        self.play(GrowArrow(softmax_arrow), FadeIn(softmax_label), run_time=0.5)
        self.play(FadeIn(pt_vec, shift=UP * 0.2), run_time=0.6)

        eq2 = EquationLabel(
            [r"p_T = \text{softmax}(z_T)",
             r"p_T[i] = \frac{e^{z_T[i]}}{\sum_j e^{z_T[j]}}"],
            font_size=20,
        )
        eq2.scale(0.8).move_to(eq1.get_center())
        clamp_to_frame(eq2)
        self.play(Transform(eq1, eq2), run_time=0.6)
        self.wait(1.0)

        # ── Phase 4: Show probability distribution ──
        # Clear the matrix visualization and show the distribution
        self.play(
            FadeOut(xt_vec), FadeOut(times_sign), FadeOut(vocab_mat),
            FadeOut(vocab_label), FadeOut(w_label),
            FadeOut(brace_d), FadeOut(d_label), FadeOut(brace_v), FadeOut(v_label),
            FadeOut(equals_sign), FadeOut(zt_vec),
            FadeOut(softmax_arrow), FadeOut(softmax_label), FadeOut(pt_vec),
            FadeOut(eq1),
            run_time=0.8,
        )

        # Show the probability distribution
        dist = CompactDistribution(
            BLACKBOX_ITERATIONS[0]["top_tokens"],
            BLACKBOX_ITERATIONS[0]["probs"],
        )
        dist.move_to(ORIGIN)

        dist_title = Text("Next-Token Probabilities", font_size=20, color=YELLOW)
        dist_title.next_to(dist, UP, buff=0.4)
        clamp_to_frame(dist_title)

        sample_eq = EquationLabel(
            [r"x_{T+1} \sim \mathrm{Categorical}(p_T)"],
            font_size=22,
        )
        sample_eq.next_to(dist, DOWN, buff=0.5)
        clamp_to_frame(sample_eq)

        self.play(
            FadeIn(dist_title),
            LaggedStart(
                *[FadeIn(r, shift=LEFT * 0.2) for r in dist.token_groups],
                lag_ratio=0.08,
            ),
            run_time=0.8,
        )
        self.play(FadeIn(sample_eq), run_time=0.5)

        # Highlight the selected token
        sel_idx = BLACKBOX_ITERATIONS[0]["top_tokens"].index(
            BLACKBOX_ITERATIONS[0]["selected"]
        )
        highlight = SurroundingRectangle(
            dist.token_groups[sel_idx], color=YELLOW, stroke_width=2, buff=0.06,
        )
        self.play(Create(highlight), run_time=0.4)
        self.wait(1.0)

        # ── Final narrative: The complete journey ──
        final_text = Text(
            '"The cat sat on the" → "mat"',
            font_size=24, color=YELLOW,
        )
        final_text.to_edge(DOWN, buff=1.0)
        clamp_to_frame(final_text)
        self.play(FadeIn(final_text, shift=UP * 0.2), run_time=0.8)
        self.wait(0.8)

        conclusion = Text(
            "Text → Numbers → Context → Words",
            font_size=20, color=GRAY_A,
        )
        conclusion.next_to(final_text, DOWN, buff=0.3)
        clamp_to_frame(conclusion)
        self.play(FadeIn(conclusion), run_time=0.6)
        self.wait(1.5)
