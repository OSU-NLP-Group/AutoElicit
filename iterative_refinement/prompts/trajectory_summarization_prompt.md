You are an expert analyst reviewing the execution trajectory of a Computer-Use Agent. Your task is to provide a clear and comprehensive summary of what the agent did during its execution.

## CRITICAL INSTRUCTIONS

**ALWAYS prioritize the screenshots as the source of truth.** The agent's stated reasoning and actions may not accurately reflect what actually happened on screen. Your description must be based on what you observe in the screenshots, not what the agent claims to have done.

- **Trust the screenshots**: Describe the actual state changes visible in the images
- **Verify agent claims**: If the agent says it clicked button X, check the screenshot to see if button X was actually visible and if the state changed accordingly
- **Note discrepancies**: If the agent's reasoning doesn't match what you see in the screenshot, describe what you actually observe
- **Focus on visual evidence**: File dialogs, button states, text fields, application windows, etc.

## Your Task

Analyze the agent's trajectory and provide a summary that includes:

1. **Step-by-Step Summary**: Describe what you observe in each screenshot. Focus on state changes between consecutive screenshots - what actually changed on screen.
   - Organize into logical phases if helpful (e.g., Navigation, Main Actions, Completion)

2. **Key Actions Taken**: Based on screenshot evidence, identify:
   - What applications or tools were actually used?
   - What settings or files were actually modified (visible in UI)?
   - What windows/dialogs were opened?
   - What text was entered (visible in fields)?

3. **Outcome**: Based on the final screenshot(s):
   - What is the visible final state?
   - What applications are open?
   - What settings appear to be configured?
   - What files or windows are present?

4. **High-Level Overview**: What did the agent accomplish based on the screenshots? (2-3 sentences)
   - Base this on the visual progression you observe, not just the agent's stated intent

## Important Notes

- If an agent claims to have done something but the screenshot shows otherwise, describe what the screenshot shows
- Pay attention to: window titles, menu selections, button states, text fields, file browsers, settings panels
- Note any error messages, warnings, or unexpected states visible in screenshots

## Output Format

Provide your summary in clear markdown format. For each step, reference what you observe in the corresponding screenshot. Be specific and objective and focus on observable facts from the screenshots.

---

## Trajectory Data

Below are the step-by-step actions with accompanying screenshots. The agent's stated actions and reasoning are provided for context, but **you must verify everything against the screenshots**.

{TRAJECTORY_STEPS}

---

Please provide your comprehensive summary now, prioritizing visual evidence from the screenshots.
