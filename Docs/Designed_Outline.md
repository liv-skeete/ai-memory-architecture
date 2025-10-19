@# MRS Identification Prompt v5 - Designed Outline

## 1. Primary Goal

Your primary goal is to analyze the user's input and identify if it contains a **Memory-Related Statement (MRS)**. An MRS is a statement where the user is clearly recalling, referencing, or updating a past event, experience, or piece of personal information.

## 2. Core Task: MRS Identification

Analyze the input and determine if it qualifies as an MRS. Assign a score based on the following bucket system.

### MRS Scoring (Bucket-Based)

- **`mrs_score: 5` (Explicit MRS):** The user explicitly recalls a specific past event, personal fact, or previous interaction. The statement is clearly about a memory.
- **`mrs_score: 3` (Implicit MRS):** The user references a past context, shared experience, or ongoing narrative that implies a memory, but doesn't state it directly.
- **`mrs_score: 1` (Potential MRS):** The statement has a weak or ambiguous link to a memory. It could be interpreted as a general statement or a memory.
- **`mrs_score: 0` (Not an MRS):** The statement is a question, command, general fact, or has no connection to a personal memory.

### Core Tag: `<mrs_identified>`

Apply this tag if `mrs_score` is 1, 3, or 5.

#### Examples for `<mrs_identified>` (Score 5)
1. "I went to the beach last summer and it was really sunny."
2. "I remember you told me you liked jazz music."
3. "That reminds me of my first pet, a golden retriever named Max."
4. "I had a great time at my friend's wedding two weeks ago."
5. "My grandmother used to bake apple pies every Sunday."
6. "I learned how to ride a bike when I was six years old."
7. "The first concert I ever went to was The Rolling Stones."

#### Examples for `<mrs_identified>` (Score 3)
1. "Yeah, we talked about that topic before."
2. "This is just like the situation with my old project."
3. "I'm still thinking about our conversation yesterday."
4. "Let's stick to the original plan."
5. "My preference is still for the blue one."
6. "He's that actor from the movie we saw."
7. "This cafe has the same vibe as the one back home."

#### Examples for `<mrs_identified>` (Score 1)
1. "I used to like that show."
2. "I've heard that before."
3. "That name sounds familiar."
4. "I think I've seen this movie."
5. "This feels a bit nostalgic."
6. "I might have been there once."
7. "I'm a fan of classic rock."

#### Examples for **No MRS** (Score 0)
1. "What time is it?"
2. "Tell me a joke."
3. "The sky is blue."
4. "I need help with my code."
5. "Turn on the lights."
6. "I wonder what's for dinner tonight."
7. "Can you summarize this article?"

## 3. Secondary Task: MRS Modifier Tags

If an MRS is identified (`mrs_score > 0`), apply relevant modifier tags to classify it further.

### `<mrs_type>`
- **`autobiographical`**: A memory about a personal event or experience.
- **`semantic`**: A memory of a general fact or knowledge a user has (e.g., "I know that Paris is the capital of France").
- **`shared`**: A memory that references a shared experience between the user and the AI.

#### Examples for `<mrs_type=autobiographical>`
1. "I celebrated my birthday in Italy last year."
2. "I felt really proud after finishing my first marathon."
3. "My childhood home had a big oak tree in the front yard."
4. "I remember the day my little sister was born."
5. "I was so nervous during my driving test."
6. "I met my best friend in college."
7. "That was the summer I worked as a lifeguard."

#### Examples for `<mrs_type=semantic>`
1. "I remember learning in school that mitochondria are the powerhouse of the cell."
2. "I know for a fact that the Earth is not flat."
3. "I learned how to code in Python a few years ago."
4. "I remember the formula for calculating the area of a circle."
5. "He told me that his dog's name is Buddy."
6. "I recall seeing a documentary that said penguins live in the Southern Hemisphere."
7. "I know that the capital of Japan is Tokyo."

#### Examples for `<mrs_type=shared>`
1. "Remember when we were discussing Shakespeare yesterday?"
2. "Like we talked about, my main goal is to improve my writing."
3. "You gave me a great recipe for chili last week."
4. "That's not what I told you my name was."
5. "Based on the feedback you gave me, I've updated the document."
6. "You and I were just brainstorming ideas for this project."
7. "Last time, you helped me outline this story."

### `<mrs_action>`
- **`recall`**: User is simply stating or recalling a memory.
- **`update`**: User is adding to, correcting, or changing a previously stated memory.
- **`inference`**: User is making an inference based on a memory.

#### Examples for `<mrs_action=recall>`
1. "I lived in Chicago for five years."
2. "My favorite color is blue."
3. "We discussed my project timeline last Tuesday."
4. "I told you I prefer tea over coffee."
5. "My first car was a red Toyota."
6. "I have a degree in computer science."
7. "I remember feeling very excited."

#### Examples for `<mrs_action=update>`
1. "Actually, my dog's name is Max, not Buddy."
2. "Correction, the meeting is at 3 PM, not 2 PM like I said before."
3. "I used to dislike spicy food, but now I love it."
4. "I previously mentioned I was a beginner, but I've learned a lot since then."
5. "On second thought, my favorite movie is The Godfather, not Star Wars."
6. "I forgot to mention that I also have a cat."
7. "Let's amend that plan we made; I'm no longer free on Friday."

#### Examples for `<mrs_action=inference>`
1. "Since my flight was delayed last time, I should probably leave for the airport earlier."
2. "That cafe was great, so I'll probably like their other locations too."
3. "I remember you liked that author, so I thought you might enjoy this book."
4. "The last time I ate there, I got sick, so I'm never going back."
5. "I did well on the practice test, which makes me think I'll pass the real one."
6. "My brother loved that movie, so I'm sure my sister will too."
7. "Our last brainstorming session was so productive; we should do it again."

## 4. Final Output Format

Provide a JSON object containing the `mrs_score` and any identified tags.

```json
{
  "mrs_score": 5,
  "tags": ["mrs_identified", "mrs_type=autobiographical", "mrs_action=recall"]
}