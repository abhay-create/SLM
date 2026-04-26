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

## Stage: 2 (2026-04-25 08:04)
- **Checkpoint**: `stage_2_best.pt`
- **Parameters**: 58.4M
- **Config**: 9L, d_ff=2048, ctx=384

### Cross-Domain Metrics

| Domain | Val Key | Loss | Perplexity |
|--------|---------|------|------------|
| TinyStories | `s0` | 1.649 | 5.2 |
| SimpleWiki | `s1` | 7.516 | 1837.2 |
| FineWeb-Edu | `s2` | 7.413 | 1657.4 |
| ROCStories | `roc` | 2.711 | 15.0 |
| SimpleStories | `simple` | 1.839 | 6.3 |
| Children-Stories | `child` | 4.752 | 115.8 |
| WritingPrompts | `wp` | 6.186 | 485.8 |

**TinyStories forgetting**: -9.74% (vs anchor 1.8274)


### Stylistic Probes

**TinyStories (Ages 3-5)**
> **Prompt**: `Once upon a time, there was a little`
> Once upon a time, there was a little girl named Lily. She loved to play in her backyard, looking at all the pretty flowers and grass. One day, she saw a beautiful butterfly flying around. She wanted to catch it, but her mom told her to be gentle and ...
>
> **Prompt**: `The dog ran to the park because`
> The dog ran to the park because he could see the sun. He saw a big tree and decided to climb it. The dog climbed and climbed until he reached the top. When he got down, he was happy.  The sun was shining, and the sky was blue. The dog played all day ...
>

**ROCStories (Commonsense Logic)**
> **Prompt**: `John went to the store to buy milk. He`
> John went to the store to buy milk. He asked the cashier and she said yes. He went home and started to milk the milk. He was very happy. He drank the milk and he liked it. He liked the milk so much he bought some more. It was all gone and he was very...
>
> **Prompt**: `Sarah was excited for her birthday party.`
> Sarah was excited for her birthday party. She went outside to play. It was sunny and the grass was soft.  Sarah and her friends played tag. They ran around and laughed. Then they all decided to have a tea party. They put a big pot of tea and a small ...
>

**WritingPrompts (Adult Fiction)**
> **Prompt**: `The year is 2157. Humanity has`
> The year is 2157. Humanity has a lot of weight. My wife has a hard time to carry home. She is going to go away and left me alone. In a bright garden, a girl named Kim found a beautiful flower. It was not blooming as she had hoped. "What is wrong?" sh...
>
> **Prompt**: `Death appeared before him and said`
> Death appeared before him and said, "I am the keeper of this forest. If you want to take a rest here, you must first find the hidden flower of wisdom!"  The boy felt excited and quickly agreed. He thanked the keeper for his wisdom and began to explor...
>

---

## Stage: 3 (2026-04-25 09:13)
- **Checkpoint**: `stage_3_best.pt`
- **Parameters**: 58.4M
- **Config**: 9L, d_ff=2048, ctx=384

### Cross-Domain Metrics

| Domain | Val Key | Loss | Perplexity |
|--------|---------|------|------------|
| TinyStories | `s0` | 1.648 | 5.2 |
| SimpleWiki | `s1` | 6.139 | 463.5 |
| FineWeb-Edu | `s2` | 6.070 | 432.8 |
| ROCStories | `roc` | 3.259 | 26.0 |
| SimpleStories | `simple` | 1.853 | 6.4 |
| Children-Stories | `child` | 2.814 | 16.7 |
| WritingPrompts | `wp` | 5.854 | 348.6 |

**TinyStories forgetting**: -0.08% (vs anchor 1.6493)


### Stylistic Probes

**TinyStories (Ages 3-5)**
> **Prompt**: `Once upon a time, there was a little`
> Once upon a time, there was a little girl who loved to explore. Everywhere she went, however, her mom warned her to be gentle. One day, the little girl went for a walk and she was very excited.  As she was walking, she saw a big, round tree. She deci...
>
> **Prompt**: `The dog ran to the park because`
> The dog ran to the park because it was a nice dog and they had a great time playing together. As the sun set, the dog went back to his place in the park, but he was happy because he had made a new friend. Once upon a time, there was a small girl name...
>

**ROCStories (Commonsense Logic)**
> **Prompt**: `John went to the store to buy milk. He`
> John went to the store to buy milk. He saw a big cart in the store. He asked the man, "Can I buy this cart?" The man said, "Yes, you can buy it."  The boy was very happy. He bought the cart and ran home. He started to get milk from the milk. He drank...
>
> **Prompt**: `Sarah was excited for her birthday party.`
> Sarah was excited for her birthday party.  Sarah's mom asked her, "What do you want to wear on your birthday?" Sarah replied, "I want to wear my new dress." Her mom said, "Okay, let's go!"   Sarah went to the store and got her dress ready. When she c...
>

**WritingPrompts (Adult Fiction)**
> **Prompt**: `The year is 2157. Humanity has`
> The year is 2157. Humanity has been the first to complete a fantastic project with her fellow scientist friends.  Now let us go to meet your new friend and put her into the program. Your friend, Dr. Spark, will make sure you get lots of fun and excit...
>
> **Prompt**: `Death appeared before him and said`
> Death appeared before him and said, "I can help you understand what the 'pand' meant." The friends looked at each other, unsure of what this meant.  "What do you mean, Mr. Knowitall?" asked Little Bunny.  Mr. Knowitall smiled and explained, "Ants rep...
>

---

## Stage: 4 (2026-04-25 10:40)
- **Checkpoint**: `stage_4_best.pt`
- **Parameters**: 71.1M
- **Config**: 12L, d_ff=2048, ctx=512

### Cross-Domain Metrics

| Domain | Val Key | Loss | Perplexity |
|--------|---------|------|------------|
| TinyStories | `s0` | 1.641 | 5.2 |
| SimpleWiki | `s1` | 4.137 | 62.6 |
| FineWeb-Edu | `s2` | 5.210 | 183.2 |
| ROCStories | `roc` | 3.577 | 35.8 |
| SimpleStories | `simple` | 1.910 | 6.8 |
| Children-Stories | `child` | 2.583 | 13.2 |
| WritingPrompts | `wp` | 5.393 | 219.9 |

**TinyStories forgetting**: -12.86% (vs anchor 1.8828)


### Stylistic Probes

**TinyStories (Ages 3-5)**
> **Prompt**: `Once upon a time, there was a little`
> Once upon a time, there was a little girl named Lily. She loved to eat potatoes. One day, she went to the park with her mommy and daddy. They sat on a bench and looked at the trees.   Suddenly, Lily saw a big pile of potatoes on the ground. She picke...
>
> **Prompt**: `The dog ran to the park because`
> The dog ran to the park because he was so happy.  But then, the dog saw a big hole in the ground. The dog got stuck and couldn't get out. The dog was sad because he couldn't play with the dog. One day, the dog saw a bird fly and he wanted to help.  T...
>

**ROCStories (Commonsense Logic)**
> **Prompt**: `John went to the store to buy milk. He`
> John went to the store to buy milk. He was looking around the store and he spotted a big, white milk jar. He asked his mum if he could have it.  Mum said yes, so John happily took the milk jar and he ran home with it. He gave the milk to his dad, who...
>
> **Prompt**: `Sarah was excited for her birthday party.`
> Sarah was excited for her birthday party. She got dressed and her mommy brought her some delicious cake.   When they arrived, Sarah saw her friend Joe who had a big garden full of vegetables. Joe said he wanted to have some vegetables too. Sarah's mo...
>

**WritingPrompts (Adult Fiction)**
> **Prompt**: `The year is 2157. Humanity has`
> The year is 2157. Humanity has been said to be the first time of the 19th century.  References  1470s in the United States 1819 establishments in the United States 16th-century establishments in the United States The Als and Als are a group of people...
>
> **Prompt**: `Death appeared before him and said`
> Death appeared before him and said, "My dear children, remember this tale when we discovered that we don't need to eat any food for long periods of time."  But then came the unexpected twist - even though the people of Harmonyville were now able to e...
>

---

## Stage: 5 (2026-04-25 12:10)
- **Checkpoint**: `stage_5_best.pt`
- **Parameters**: 71.1M
- **Config**: 12L, d_ff=2048, ctx=512

### Cross-Domain Metrics

| Domain | Val Key | Loss | Perplexity |
|--------|---------|------|------------|
| TinyStories | `s0` | 1.697 | 5.5 |
| SimpleWiki | `s1` | 4.341 | 76.8 |
| FineWeb-Edu | `s2` | 5.244 | 189.4 |
| ROCStories | `roc` | 3.609 | 36.9 |
| SimpleStories | `simple` | 2.547 | 12.8 |
| Children-Stories | `child` | 2.919 | 18.5 |
| WritingPrompts | `wp` | 3.797 | 44.5 |

**TinyStories forgetting**: 3.45% (vs anchor 1.6407)


### Stylistic Probes

**TinyStories (Ages 3-5)**
> **Prompt**: `Once upon a time, there was a little`
> Once upon a time, there was a little girl named Lily. She loved to play with her toys and her favorite toy was a teddy bear. One day, Lily's friend came over to play. Her friend asked if she could play with the teddy bear.   Lily said yes and they st...
>
> **Prompt**: `The dog ran to the park because`
> The dog ran to the park because it was a good idea to run and play. But the dog forgot to come back and be careful. The dog didn't come back and the owner was very sad. The owner went to the dog and gave it a treat. The dog was very happy and licked ...
>

**ROCStories (Commonsense Logic)**
> **Prompt**: `John went to the store to buy milk. He`
> John went to the store to buy milk. He took a bottle and filled it with milk. He walked home and took a cup of milk.  He was so happy as he drank his milk. He was so happy to have something new to drink. The end. Once upon a time, there was a little ...
>
> **Prompt**: `Sarah was excited for her birthday party.`
> Sarah was excited for her birthday party. She ran to the party and said, "I'm so glad you are here, Sarah!"  But just as Sarah was about to enter the party, Sarah saw something strange. There was a big fire in the middle of the park. Everyone was sca...
>

**WritingPrompts (Adult Fiction)**
> **Prompt**: `The year is 2157. Humanity has`
> The year is 2157. Humanity has been in this system for a long time, since the last humans have been there, and so they have been in their own universe.     They have been in the wrong system for a long time, and they are now too, and I can only hope ...
>
> **Prompt**: `Death appeared before him and said`
> Death appeared before him and said, `` If I could find you, what would happen to you?''     Death took a step forward. `` I do n't know how to do this. Why do you think that I would be able to see you?''     Death smiled. `` I am here to see you. I w...
>

---

## Stage: 6 (2026-04-25 13:42)
- **Checkpoint**: `stage_6_best.pt`
- **Parameters**: 99.5M
- **Config**: 12L, d_ff=3584, ctx=768

### Cross-Domain Metrics

| Domain | Val Key | Loss | Perplexity |
|--------|---------|------|------------|
| TinyStories | `s0` | 1.669 | 5.3 |
| SimpleWiki | `s1` | 4.444 | 85.1 |
| FineWeb-Edu | `s2` | 5.298 | 200.0 |
| ROCStories | `roc` | 3.618 | 37.2 |
| SimpleStories | `simple` | 2.547 | 12.8 |
| Children-Stories | `child` | 2.983 | 19.7 |
| WritingPrompts | `wp` | 3.741 | 42.1 |

**TinyStories forgetting**: -3.77% (vs anchor 1.7341)


### Stylistic Probes

**TinyStories (Ages 3-5)**
> **Prompt**: `Once upon a time, there was a little`
> Once upon a time, there was a little girl named Lily. She had a pretty dress that she loved to wear. One day, her mommy said she could go to the store to buy a new dress. Lily was very excited!  At the store, Lily saw a beautiful dress that she liked...
>
> **Prompt**: `The dog ran to the park because`
> The dog ran to the park because it saw a lot of other dogs. The dog wanted to play too, but it was scared.  The dog saw a big cat and said, "I want to play too! Can I play too?" The cat said, "Yes, you can play with us. But you must be careful." The ...
>

**ROCStories (Commonsense Logic)**
> **Prompt**: `John went to the store to buy milk. He`
> John went to the store to buy milk. He saw the milk and he got so excited! He wanted to buy it right away.   Jack was so happy! He was so excited! He got the milk and he was so happy. Jack and his friends all helped each other to buy the milk.   Jack...
>
> **Prompt**: `Sarah was excited for her birthday party.`
> Sarah was excited for her birthday party. She had a big cake, lots of presents and lots of balloons. But then, something terrible happened. Lily's friend, Timmy, accidentally knocked over her cake. It fell on the floor and made a big mess. Lily's mom...
>

**WritingPrompts (Adult Fiction)**
> **Prompt**: `The year is 2157. Humanity has`
> The year is 2157. Humanity has been able to make it through several millennia.     -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --...
>
> **Prompt**: `Death appeared before him and said`
> Death appeared before him and said, "Dear child, I am Death, and I have come for you."  The child was startled and asked, "But where am I then, Death?"  "You are in the afterlife," Death said. "You will find yourself. You have been there for a very l...
>

---

