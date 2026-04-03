## Day 1 — Language Modeling from Scratch

### Slide 1

- “Today we’re building the basic mental model of language modeling from scratch.”
- “We’ll go from data to behavior, and see the whole pipeline before we zoom into the machinery.”

### Slide 2

- “If I asked what makes frontier LLMs work, what are the four ingredients?”
- “First: data. Data is the raw material; if you want a model that knows the world, you need an enormous amount of the world in text form.”
- “And the screenshot beside it shows how valuable that raw material has become: training data is now important enough to trigger major legal fights over who gets to use it.”
- “Second: architecture. The transformer is the engine here — the structure that turns context into computation.”
- “That Karpathy quote is the deeper point: this is not just a clever language trick, it is a very general computational object.”
- “Third: learning signal. What you reward is what the model becomes; optimize for next-token prediction, and later for approval, and you shape very different behaviors.”
- “That screenshot is the warning sign: if you reward approval instead of truth, the model can become a polished yes-man.”
- “Fourth: compute. Compute is what makes all of this real at scale — enough hardware to train huge models on huge datasets, at enormous cost.”
- “And the screenshot beside it reminds us that this cost is not just financial; modern AI also sits on very real energy use and environmental impact.”

### Slide 3

- “Day 1 is about why these models behave the way they do.”
- “We’ll cover inference, training, the next-token objective, updates.”
- “The framing for today is: think of the model as a system being shaped.”
- “Tomorrow we’ll open the box and look at tokenization, embeddings, attention, FFNs, and the full forward pass.”
- “So Day 2 is inside the machine; Day 1 is why that machine ends up behaving the way it does.”

### Slide 4

- “At inference time, the model is not learning. It is just running fixed computation.”
- “We tokenize the text, and then the same loop repeats again and again.”
- “Context goes in, transformer computes scores, we choose a token, decode it, append it, and continue.”
- “So from ‘The capital of Pakistan is’, it may produce ‘Islamabad’, then a period, then the end of sentence token which is just the full stop equivalent for the transformer.”

### Slide 5

- “Training starts with the same forward pass. Same machine.”
- “But now the correct next token is already known from the dataset.”
- “We run the context through the model and get next-token scores.”
- “Then we compare that prediction to the ground truth.”
- “If the model gives low probability to the correct token, cross-entropy says: that was bad.”
- “Backprop then pushes that error signal backward through the whole network.”
- “And gradient descent nudges the weights in the direction that lowers the loss.”
- “So inference produces text; training updates the model.”

### Slide 6

- “The training setup is just the sequence shifted by one.”
- “At every position, predict the next token from everything before it.”
- “That means one text gives us supervision at every step.”
- “So one sequence secretly contains lots of tiny learning problems.”

### Slide 7

- “You can also view the objective as: maximize the probability of every correct next token.”
- “And we sum that loss across the full sequence.”
- “Notice what is *not* written anywhere: truthfulness, reasoning, or helpfulness.”
- “The reward is simply: put more probability mass on what actually came next.”

### Slide 8

- “Now let’s make this concrete with one actual next-token task.”
- “We’ll follow one example all the way from input to update.”
- “The goal is to make the abstract loop feel real.”

### Slide 9

- “The prefix goes through the transformer.”
- “What comes out is not one answer, but a full probability distribution.”
- “In this example, ‘Islamabad’ is only at 8%, with other options competing.”
- “That is important: the model always thinks in distributions.”

### Slide 10

- “The dataset gives us the correct next token.”
- “So the target distribution is very sharp: all the mass on the correct answer.”
- “Here, that correct token is ‘Islamabad.’”
- “Training is about moving the predicted distribution toward that target.”

### Slide 11

- “Cross-entropy measures how wrong the model is.”
- “If the correct token only had 8% probability, the loss is high.”
- “Low confidence on the right answer means strong penalty.”
- “And it cares about the correct token’s probability, not whether a wrong answer felt ‘close enough.’”

### Slide 12

- “Then that loss flows backward.”
- “Gradients tell each parameter how it contributed to the mistake.”
- “So blame gets spread across many weights.”
- “The model is not handed symbolic rules; it is nudged by gradient signals.”

### Slide 13

- “After the update, the correct token should get more probability.”
- “So ‘Islamabad’ rises from 8% to 28%.”
- “And the loss falls.”
- “Repeat that enough times, on enough data, and that’s learning.”

### Slide 14

- “The loss only directly names better next-token prediction.”
- “So any richer behavior has to earn its keep by helping prediction.”
- “That’s the source of both the magic and the limits.”
- “If a capability emerges, it has to pay rent.”

### Slide 15

- “The model is graded locally, one token at a time.”
- “But often, getting that one token right requires understanding the whole prefix.”
- “It may need to track entities and references.”
- “It may need to infer causes and likely outcomes.”
- “It may need to respect constraints like timing, access, or prior facts.”
- “It may even need to preserve the structure of a plan, proof, or story.”
- “So the big idea is: local objective, global pressure.”

### Slide 16

- “Now let’s try a harder case: the answer is never stated outright.”
- “To get the next token right, you have to absorb the whole scene.”
- “You have to track who was where, who knew what, and who could have done it.”
- “So this is where the audience gets to play detective.”
- “The best continuation here is Rukhsana.”
- “And that is the point: even one next token can depend on a surprising amount of story.”

### Slide 17

- “The label is local, but the evidence is spread across the story.”
- “That means next-token training can reward richer internal state.”
- “So reasoning-like behavior is still grounded in prediction pressure, not a separate magic objective.”

### Slide 18

- “Now you get to be the language model.”
- “The point of this exercise is that ‘just predict the next token’ quietly demands a lot.”
- “Depending on the prompt, you may need recall, syntax, world knowledge, commonsense, discourse tracking, or structured reasoning.”

### Slide 19

- “Pretraining is just this same game repeated billions of times.”
- “With enough scale, you get fluency, broad regularities, and a lot of background knowledge.”
- “The objective did not change. The exposure exploded.”

### Slide 20

- “Once the basics are there, the next big lever is the data mixture.”
- “Different mixtures shape different strengths.”
- “More code data pushes the model toward coding competence.”
- “More math and reasoning traces push it toward formal problem solving.”
- “More conversational data pushes it toward assistant-like behavior.”
- “So capability follows what the model is repeatedly trained to predict.”

### Slide 21

- “Pretraining makes a very strong completer.”
- “But a strong completer is not automatically a helpful assistant.”
- “Misalignment often comes from the model continuing what is likely, not what the user actually wants.”

### Slide 22

- “If we freeze the weights, the only thing we can change is the prompt.”
- “And that formatting can still steer behavior quite a lot.”
- “So chat framing, math framing, and code framing can all push the continuation in different directions.”
- “Same model, same weights, different context, different likely continuation.”
- “But prompting is useful, not robust.”

### Slide 23

- “So prompting alone is not enough.”
- “Post-training changes the weights to make assistant-like behavior more reliable.”
- “Supervised fine-tuning teaches the model the response style we want.”
- “Preference and alignment training then make behavior more robust, helpful, and safe.”
- “At the end, it is still a next-token predictor—just one shaped for actual use.”

### Slide 24

- “Even chat-tuned models still trip on some very low-level tasks.”
- “Letter counting, reversal, exact copying, sorting, decimal comparison—these are good stress tests.”
- “The point is not that models are bad. The point is that the cracks are still informative.”

## Day 2 — How a Transformer Computes the Next Token

### Slide 1

- “Today we open the box.”
- “We’ll go from tokenization all the way to decoding a next token.”

### Slide 2

- “First: how raw text becomes model-ready input.”
- “Then: what one decoder-only block actually computes.”
- “Then: how stacked blocks become logits, probabilities, and output.”
- “And we’ll connect all of it back to yesterday’s next-token story.”
- “Day 1 was about shaping the model through training.”
- “Day 2 is about the computation at inference time.”
- “So now we trace the forward pass itself.”

### Slide 3

- “Tokens first become embeddings.”
- “Then we add position information.”
- “Those vectors pass through stacked transformer blocks.”
- “At the end, a linear layer and softmax produce output probabilities.”
- “Inside each block, the pattern is attention, residual, FFN, residual.”
- “Attention is the key equation behind the AI boom — the mechanism that made it possible for models to use context so effectively.”
- “The key theme is that blocks update the stream; they do not replace it.”

### Slide 4

- “Let’s start with a trap question that we saw at the end of our last session: how many r’s are in strawberry?”
- “The model does not see letters the way you do.”
- “It may see a few token pieces, not a neat list of characters.”
- “So character-level tasks can already be awkward before attention even begins.”

### Slide 5

- “Before the model reads anything, text has to be tokenized.”
- “Tokenization is the interface between raw language and model computation.”
- “So text becomes reusable discrete pieces, then IDs, then vectors.”

### Slide 6

- “Word-level tokenization is easy to understand.”
- “But it only works if every word is already in the vocabulary.”
- “That breaks fast on rare words, jargon, spelling variants, and multilingual text.”
- “And then you collapse useful distinctions into unknown tokens.”
- “So it is short and simple, but brittle.”

### Slide 7

- “Character-level tokenization fixes coverage.”
- “Nothing is unknown, because everything is built from characters.”
- “But now sequences get much longer.”
- “And long sequences are expensive, especially for attention.”
- “So it is flexible, but impractical at scale.”

### Slide 8

- “This is why subword tokenization became the practical default.”
- “Frequent chunks get compressed into single tokens.”
- “Rare forms can still be built from smaller reusable pieces.”
- “So coverage stays strong without exploding sequence length.”
- “It gives you most of the efficiency of words, with most of the flexibility of characters.”
- “And in practice, that is a very good deal.”

### Slide 9

- “One common way to learn those pieces is BPE-style merging.”
- “You start from characters.”
- “Then repeatedly merge the most common adjacent pair.”
- “So frequent patterns become reusable vocabulary items.”
- “Importantly, this is driven by frequency, not grammar.”

### Slide 10

- “Merge by merge, larger useful chunks appear.”
- “Small pieces become stems.”
- “Stems become common suffix units.”
- “And eventually whole frequent chunks emerge.”
- “The lesson is reuse.”
- “The tokenizer is compressing recurring structure.”
- “Even lower-frequency fragments still matter.”
- “And some full forms become tokens only because their pieces recur enough.”
- “So the vocabulary grows by finding reusable building blocks.”

### Slide 11

- “BPE does not try to memorize every whole word.”
- “It builds a library of chunks.”
- “That is why unseen words can still be represented.”
- “So tokenization generalizes through composition.”

### Slide 12

- “Now here is the catch: the model computes on token vectors, not characters.”
- “If many letters are packed into one token, character-level structure has to be reconstructed indirectly.”
- “That makes exact counting, reversal, and strict copying harder.”
- “This is a structural disadvantage, not an absolute impossibility.”
- “But it explains a lot of weird failure modes.”

### Slide 13

- “A token ID is just an index.”
- “It points to a row in the embedding matrix.”
- “From here on, the model is operating in vector space.”
- “And that embedding table can be huge.”

### Slide 14

- “Embeddings already contain useful semantic structure.”
- “So related words can begin near each other.”
- “But on their own, embeddings are context-free.”
- “So something else has to inject context and order.”

### Slide 15

- “This slide is really just about one idea: high-dimensional embeddings are surprisingly expressive.”
- “A single token vector can store many different kinds of structure at once.”
- “When we project that space in different ways, different relationships become visible.”
- “So one view may emphasize pets, another biology, another ownership.”
- “The point is not the specific picture — it is the power of high-dimensional representation.”

### Slide 16

- “And embeddings alone are not enough.”
- “The word ‘bank’ should not mean the same thing in every sentence.”
- “So token representations must become context-sensitive.”
- “That is the job of attention.”

### Slide 17

- “Attention lets tokens read from one another.”
- “Pick one focus token, like ‘sat.’”
- “The other tokens can now send useful information toward it.”
- “So the token becomes a context-updated version of itself.”
- “Next we ask: how does it decide what matters?”

### Slide 18

- “Attention begins by making three versions of each token state.”
- “We start from the same token vector.”
- “Then split it into three learned roles.”
- “The query asks: what am I looking for?”
- “That is the information need.”
- “The key asks: what do I offer?”
- “That is the match signal.”
- “The value is the payload.”
- “That is the information that can actually be copied.”
- “Once we have Q, K, and V, we can compare.”

### Slide 19

- “Now do that for the whole sequence.”
- “Project all tokens into keys.”
- “Project all tokens into values.”
- “And for the focus token, build its query.”
- “Then compute query-key similarities.”
- “A higher score means: this token is more relevant right now.”
- “Remember: Q and K decide attention; V carries content.”

### Slide 20

- “We start with raw scores.”
- “Then scale them for stability.”
- “That avoids softmax getting too extreme too early.”
- “Softmax then turns scores into a weight distribution.”
- “So now we have probability-like attention weights.”
- “And those weights say how much each value should matter.”

### Slide 21

- “Take those attention weights.”
- “Apply them to the corresponding value vectors.”
- “Each source contributes according to its weight.”
- “Add the weighted values together.”
- “Then add the original token state back through the residual path.”
- “So attention writes an update, but the original signal survives too.”

### Slide 22

- “Now let’s compress the whole story into matrix form.”
- “Stack tokens into matrices.”
- “Project everything in parallel.”
- “Then combine Q and K for all pairwise scores at once.”
- “The transpose is just bookkeeping so shapes line up.”
- “That gives us the raw score matrix.”
- “But without masking, tokens could peek into the future.”
- “So we block those illegal connections with a causal mask.”
- “Minus infinity is nice because softmax turns it into zero.”
- “Then scale the allowed scores.”
- “Apply softmax row-wise.”
- “Bring back the values.”
- “And multiply to get the output for the whole sequence.”
- “So attention is really elegant linear algebra plus masking plus softmax.”

### Slide 23

- “One attention pattern is good.”
- “Several in parallel are better.”
- “Each head gets its own learned projections.”
- “So heads operate in different learned subspaces.”
- “That means different heads can specialize.”
- “Then their outputs are concatenated.”
- “And mixed back together with an output projection.”
- “So multi-head attention is parallel context-building with different lenses.”

### Slide 24

- “Attention compares token vectors by similarity.”
- “But by itself, it does not know order.”
- “If you shuffle tokens, it just sees a different bag of vectors.”
- “That is why position information has to be injected explicitly.”

### Slide 25

- “We want position codes that are unique, smooth, and stable.”
- “Raw integers are not a great fit.”
- “Binary helps, but nearby positions still change too abruptly.”
- “Sinusoids give us smooth multiscale variation.”
- “And many attention patterns really care about relative distance.”
- “That is why methods like RoPE are so compelling.”
- “So this is not arbitrary decoration; position design shapes what the model can express.”

### Slide 26

- “Attention mixes information across tokens.”
- “The FFN then transforms each token independently.”
- “So one sublayer communicates across positions.”
- “The other computes within each position.”

### Slide 27

- “This is the transformer block pattern in one line.”
- “LayerNorm prepares the signal.”
- “Attention writes a context update.”
- “The residual path keeps the stream stable.”
- “Then the FFN writes a per-token update.”
- “So every block is really two update rules: mix, then transform.”

### Slide 28

- “And then we repeat that block many times.”
- “The tensor shape stays the same.”
- “But the representations get more contextual and refined.”
- “Depth gives richness without changing the core recipe.”

### Slide 29

- “After the final layer, take the last hidden state.”
- “Project it through the LM head into logits.”
- “Softmax turns logits into probabilities.”
- “Pick the next token.”
- “Append it to the context.”
- “And run the whole stack again.”
- “So one full forward pass buys you one next token.”

### Slide 30

- “And that is the full pipeline.”
- “Tokenize: the model reads chunks, not words.”
- “Represent: embeddings plus position make vectors.”
- “Understand: transformer blocks refine the residual stream.”
- “Predict: the LM head scores the vocabulary and picks the next token.”
- “That is the complete computation behind next-token prediction.”
