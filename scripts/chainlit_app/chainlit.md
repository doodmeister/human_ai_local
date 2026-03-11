# George — Cognitive Chat

Welcome to **George**, a cognitive AI assistant with working memory, long-term memory, attention, executive planning, and learning.

## What can you do here?

**Chat** — just type a message. George retrieves relevant memories, applies attention/salience scoring, and replies in context.

**Commands** — click the `/` icon or type:

| Command | Description |
|---------|-------------|
| `/memory stm` | Browse short-term memory |
| `/memory ltm` | Browse long-term memory |
| `/memory semantic` | Browse semantic facts |
| `/reminders` | List active reminders |
| `/remind 30 Review PR` | Set a 30-minute reminder |
| `/goals` | List active goals |
| `/goal Finish the report` | Create a new goal |
| `/dream` | Run STM → LTM consolidation |
| `/reflect` | Metacognitive self-analysis |
| `/learning` | ML learning dashboard |

**Settings** — click the ⚙️ icon to change LLM provider, model, salience threshold, and toggle memory/attention.

> Make sure the API backend is running with `python main.py api`, or launch Chainlit with `python main.py chainlit --with-backend`.
