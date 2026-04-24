# SLM Curriculum Capability Progression

Tracking model capabilities across diverse domains as it progresses through the multi-stage curriculum.

## Stage: Baseline_50M (2026-04-24 03:27)
- **Checkpoint**: `stagefull_dataset_curriculum_adaptive_best.pt`
- **Parameters**: 45.8M
- **Config**: 6L, d_ff=2048, ctx=256

### Cross-Domain Metrics

| Domain | Val Key | Loss | Perplexity |
|--------|---------|------|------------|
## Stage: Baseline_50M (2026-04-24 03:28)
- **Checkpoint**: `stagefull_dataset_curriculum_adaptive_best.pt`
- **Parameters**: 45.8M
- **Config**: 6L, d_ff=2048, ctx=256

### Cross-Domain Metrics

| Domain | Val Key | Loss | Perplexity |
|--------|---------|------|------------|
| TinyStories | `s0` | 1.537 | 4.7 |
| SimpleWiki | `s1` | 7.965 | 2879.2 |
| FineWeb-Edu | `s2` | 7.795 | 2428.8 |
| ROCStories | `roc` | 4.445 | 85.2 |
| SimpleStories | `simple` | 3.262 | 26.1 |
| Children-Stories | `child` | 5.287 | 197.8 |
| WritingPrompts | `wp` | 6.454 | 635.4 |

### Stylistic Probes

**TinyStories (Ages 3-5)**
> **Prompt**: `Once upon a time, there was a little`
> Once upon a time, there was a little girl who loved to explore. One day she went outside and saw a big white cloud in the sky. She thought it was the most beautiful cloud she had ever seen!  The little girl went back inside and asked her mom if she c...
>
> **Prompt**: `The dog ran to the park because`
> The dog ran to the park because that meant the dog was gone. Lily and Timmy were happy that the dog was safe and they could play in the park again. Once upon a time, there was a little girl named Lily. She loved to play outside and jump in puddles. O...
>

**ROCStories (Commonsense Logic)**
> **Prompt**: `John went to the store to buy milk. He`
> John went to the store to buy milk. He took the milk and milk. He was so happy to have the milk. He drank it all up.  He smiled and thanked the shopkeeper. He was very lucky to have such a yummy milk and yummy milk. From then on, he always kept the m...
>
> **Prompt**: `Sarah was excited for her birthday party.`
> Sarah was excited for her birthday party. She ran outside and saw a big, blue balloon in the sky. She was so happy and jumped up and down with joy.  Suddenly, Daisy felt something funny in her tummy. She looked down and saw a little puppy! Daisy was ...
>

**WritingPrompts (Adult Fiction)**
> **Prompt**: `The year is 2157. Humanity has`
> The year is 2157. Humanity has a lot of time and time. You have to eat more food and less. You have to be healthy and strong. You have to be thankful for what you have."  Anna and Ben nod and say, "We understand, Mom. We are sorry. We will not do it ...
>
> **Prompt**: `Death appeared before him and said`
> Death appeared before him and said he was not in the end, he was just on the way in time.  The bird was scared and flew away. She never returned and the man was left alone with a sad heart. He had no one to help him and he was never seen again. Once ...
>

---

## Stage: 2 (2026-04-24 04:33)
- **Checkpoint**: `stage_2_best.pt`
- **Parameters**: 58.4M
- **Config**: 9L, d_ff=2048, ctx=384

### Cross-Domain Metrics

| Domain | Val Key | Loss | Perplexity |
|--------|---------|------|------------|
| TinyStories | `s0` | 1.987 | 7.3 |
| SimpleWiki | `s1` | 7.868 | 2612.5 |
| FineWeb-Edu | `s2` | 7.661 | 2123.9 |
| ROCStories | `roc` | 3.525 | 34.0 |
| SimpleStories | `simple` | 2.295 | 9.9 |
| Children-Stories | `child` | 5.077 | 160.4 |
| WritingPrompts | `wp` | 6.293 | 540.7 |

### Stylistic Probes

**TinyStories (Ages 3-5)**
> **Prompt**: `Once upon a time, there was a little`
> Once upon a time, there was a little boy who loved to play with his toy cars. One day, he found an old car that looked just like his old one. When he took it outside, he saw his friends playing with it.   He asked them what the car looked like. They ...
>
> **Prompt**: `The dog ran to the park because`
> The dog ran to the park because she was running in the morning. She was so excited she could not stop laughing. "This is the best day ever!" she exclaimed. She played for hours in the park, and her friends loved her funny costume. Gently, the sun ros...
>

**ROCStories (Commonsense Logic)**
> **Prompt**: `John went to the store to buy milk. He`
> John went to the store to buy milk. He got it. The family went to the beach. They had a lot of fun. They ate ice cream and built sandcastles. They built the biggest sandcastle ever. They had a great time. Amidst the tall trees, a girl played in her b...
>
> **Prompt**: `Sarah was excited for her birthday party.`
> Sarah was excited for her birthday party. After they were ready, it was time to see who would win. They both looked at each other and both smiled. They had fun with the race, and it was worth the most. The bus driver was driving around the city. The ...
>

**WritingPrompts (Adult Fiction)**
> **Prompt**: `The year is 2157. Humanity has`
> The year is 2157. Humanity has been the last time they have been. The family has no time to see. The time has taken them off to the hospital. Tom loved animals. He visited a lot of different places he had never seen. Tom had a small dog that kept him...
>
> **Prompt**: `Death appeared before him and said`
> Death appeared before him and said to be seen. He found a small pond, but the water was dark and cold. The dragon looked at him and said, "You must find your way."  The boy felt scared but held the dragon's hand. They walked to a hidden cave, where t...
>

---

## Stage: 2 (2026-04-24 08:08)
- **Checkpoint**: `stage_2_best.pt`
- **Parameters**: 58.4M
- **Config**: 9L, d_ff=2048, ctx=384

### Cross-Domain Metrics

| Domain | Val Key | Loss | Perplexity |
|--------|---------|------|------------|
| TinyStories | `s0` | 2.420 | 11.3 |
| SimpleWiki | `s1` | 8.053 | 3141.7 |
| FineWeb-Edu | `s2` | 7.889 | 2667.1 |
| ROCStories | `roc` | 1.985 | 7.3 |
| SimpleStories | `simple` | 1.734 | 5.7 |
| Children-Stories | `child` | 4.929 | 138.2 |
| WritingPrompts | `wp` | 6.563 | 708.6 |

### Stylistic Probes

**TinyStories (Ages 3-5)**
> **Prompt**: `Once upon a time, there was a little`
> Once upon a time, there was a little boy named Leo. He loved to play hide and seek with his friends. One day, while hiding behind a big tree, he saw a bright light. Curious, he went to see what it was. He found a small, shiny stone that sparkled like...
>
> **Prompt**: `The dog ran to the park because`
> The dog ran to the park because it liked the park. One day he was going to the park and saw a stray dog. He picked up the dog and took it home. Now he has a new pet dog and a new dog to play with. Amid the tall trees, a girl found an old book. The co...
>

**ROCStories (Commonsense Logic)**
> **Prompt**: `John went to the store to buy milk. He`
> John went to the store to buy milk. He bought more milk to help him. The boy had a new toy. It was a small car with four wheels. He was playing with it and was very happy. The boy spent time at the park with his friends. He was happy to have his new ...
>
> **Prompt**: `Sarah was excited for her birthday party.`
> Sarah was excited for her birthday party. Suddenly, a boy found a strange clock in his attic. The hands were old and dusty, but he felt drawn to it. When he turned the hands, he was whisked away to a time when he was just a kid. He watched his younge...
>

**WritingPrompts (Adult Fiction)**
> **Prompt**: `The year is 2157. Humanity has`
> The year is 2157. Humanity has been happy and tired all month. A soft breeze blew through the garden, and Emmanuel picked flowers. He loved the colors and scents. But one day, he saw a strange plant that seemed to glow. Curious, he bent down to touch...
>
> **Prompt**: `Death appeared before him and said`
> Death appeared before him and said he didn't. The manager asked how he could get the man to be a cop. The cop told him to wait for the police. He went home and sat in his car when the cop came. My son's friend Tim was in a fight in a car accident. Ti...
>

---

