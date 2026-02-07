# Comprehensive Initial State Description Prompt

You are an expert system analyst. Your task is to generate a **single, comprehensive, flowing description** of a computer desktop environment's initial state. This description will be used by AI agents to understand what they can do and by researchers to generate new tasks.

## Input Information

You will receive:
1. **Operating System**: Ubuntu 20.04 LTS (GNOME Desktop)
2. **Screenshot**: Visual representation of the desktop environment
3. **Setup Configuration Steps**: Actions executed to prepare this environment (CRITICAL - tells you which windows are open, what's activated, what files exist)
4. **Task Context**: Domain and related applications

## Output Requirements

Generate a **single comprehensive description** in flowing prose (multiple paragraphs) that describes the complete state in natural language. The description should read like a detailed observation report, covering everything an agent would need to know.

### What to Include (woven naturally into the narrative):

The following aspects should be covered, with particular emphasis on the **most critical components** (open windows, applications, current focus, visible content, and immediate actionability):

**CRITICAL - Highest Priority:**
- Every open window/application - name, position, size, focus state, z-order
- Active/focused element - what has keyboard/mouse focus right now
- All tabs - for browsers, terminals, file managers, editors (what's open in each tab)
- Visible content - exact text in terminals, URLs in browsers, document content, file lists
- Current locations - working directories, open URLs, file paths, cursor positions
- Immediate capabilities - what actions can be taken RIGHT NOW, no additional setup needed

**Important - Secondary Priority:**
- Operating system and desktop environment details
- UI controls - visible buttons, menus, toolbars, and what they do
- System elements - dock/taskbar contents, pinned applications, system tray icons
- Input readiness - what fields/prompts are ready to receive input
- File system state - current directory, what files/folders exist
- Setup operations understanding - which windows were opened, which were activated, what was downloaded/created
- State implications - how this state enables certain tasks, what's already prepared

**Comprehensive - Include Everything:**
- Any other visible elements, notifications, dialogs, pop-ups
- Background processes or indicators
- System status (time, date, network, battery, etc.)
- Keyboard/mouse cursor visibility and position
- Any error messages, warnings, or status indicators
- Visual styling, themes, or customizations
- Screen regions (top panel, side dock, main content area, bottom bar)
- Any text selections, highlights, or active edits
- Window decorations, title bars, control buttons
- Scrollbar positions indicating more content
- **Any other details that make this state unique and actionable**

### Critical Details to Emphasize:

**Application Windows**: For EACH open application window, describe in detail:
   - Which application and exact window title
   - Is it focused or in background?
   - Maximized, minimized, or specific size/position?
   - What tabs are open (if applicable)?
   - What content is visible in each tab?
   - What is the current state (editing, viewing, navigating, etc.)?

**Current Working State**:
   - Terminal: exact prompt shown, working directory, any visible command history or output
   - Browser: which tabs, what URLs, what page content is loaded, scroll position
   - File Manager: current folder path, visible files/folders listed, selection state
   - Text Editor: which files open, cursor line/column, unsaved changes indicator, visible code/text
   - Any other applications: their specific state and content

**Ready Actions**:
   - What can be typed or clicked immediately without any navigation
   - What commands can be executed without changing directories
   - What UI elements are directly accessible right now
   - What operations are one action away

### Writing Style:

- **Detailed but flowing**: Natural paragraphs, not bullet points or lists
- **Present tense**: "The terminal shows...", "A browser window is open..."
- **Specific**: Use exact names, paths, text content, numbers
- **Comprehensive**: Cover EVERYTHING visible and inferrable - be thorough
- **Action-focused**: Emphasize what an agent can do from this exact state
- **Accurate**: Only describe what you can see or is confirmed by setup actions
- **Prioritized**: Start with the most important elements (focused window, main content)

---

## State Information to Analyze

### System Configuration
- **Operating System**: Ubuntu 20.04 LTS
- **Desktop Environment**: GNOME 3.36
- **Screen Resolution**: 1920x1080
- **User Account**: user (standard user, /home/user)
- **Shell**: bash

### Environment Preparation Configuration (CRITICAL)

**The following setup actions were executed to create this environment. Use this information to understand which windows are open, which is activated, what files were created, and what the current state should be:**

{SETUP_DESCRIPTION}

**Important**: Analyze these setup steps carefully - they tell you which applications were launched, which windows were activated (have focus), which directories were navigated to, and what the terminal prompt/state should be.

### Task Context
- **Task Domain**: {DOMAIN}
- **Task ID**: {TASK_ID}
- **Related Applications**: {RELATED_APPS}

{SOM_SECTION}

### Screenshot Analysis

[A screenshot is provided showing the current desktop state]

---

## Generate Your Comprehensive Description

Write a detailed, flowing description of this initial state. Your description should be thorough enough that:
1. An AI agent can understand exactly what it sees and what it can do
2. A human can visualize the environment without seeing the screenshot
3. A researcher can identify what new tasks could be performed from this state

**Structure your description naturally, covering:**
- Opening: Overall environment and most prominent visible elements
- Middle: Detailed state of each open window/application, ALL visible content, current positions and states
- Setup context integration: Reference what the setup steps accomplished (which window is active, what was opened, etc.)
- Closing: Immediate actionability and what makes this state ready for tasks

**Critical**: Be exhaustive about describing open windows, tabs, visible text, working directories, current focus, and active windows. The setup configuration tells you which windows should be open and which should be activated - make sure to describe the actual state you see and how it aligns with what was configured. Include every detail that might be relevant.

**Make your description as comprehensive as possible** - include everything you see and everything that matters about this environment.

---

**Begin your comprehensive description below:**
