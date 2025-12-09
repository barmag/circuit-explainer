# BMAD-METHOD Official Installation Guide
## Verified Documentation for Circuit-Explainer Project

Based on official BMAD-METHOD v6 Alpha documentation.

---

## Installation

### Official Installation Command

```bash
# Install BMAD v6 Alpha (recommended for new projects)
npx bmad-method@alpha install

# Alternative: Install stable v4 (production)
npx bmad-method install
```

### What the Installer Does

The interactive installer will guide you through:

1. **Project location selection** - Where to install BMAD
2. **Module selection** - Choose which modules to install:
   - **BMM** (BMad Method) - Core agile development framework
   - **BMB** (BMad Builder) - Create custom agents/workflows
   - **CIS** (Creative Intelligence Suite) - Shared capabilities
3. **Configuration** - Project name, language preferences
4. **Optional features** - Additional capabilities (e.g., game development for BMM)
5. **IDE integration** - Set up your development environment

### Created Directory Structure

After installation, BMAD creates a unified structure:

```
your-project/
└── bmad/
    ├── core/         # Core framework (always installed)
    ├── bmm/          # BMad Method (if selected)
    │   ├── agents/   # Pre-built agents for software development
    │   ├── workflows/ # Standard workflows (feature, refactor, bugfix)
    │   └── config.yaml
    ├── bmb/          # BMad Builder (if selected)
    │   ├── workflows/ # Tools to create custom agents/workflows
    │   │   ├── create-agent/
    │   │   ├── create-workflow/
    │   │   └── create-module/
    │   └── config.yaml
    ├── cis/          # Creative Intelligence Suite (shared)
    └── _cfg/         # Your customizations
        └── agents/   # Agent customization files
```

---

## Installation for Circuit-Explainer

### Step 1: Run Installer

```bash
cd d:/code/circuit-explainer
npx bmad-method@alpha install
```

### Step 2: Answer Installer Prompts

**Recommended selections for your project:**

```
? Where would you like to install BMAD?
  › Current directory (d:/code/circuit-explainer)

? Which modules would you like to install?
  › [x] BMM (BMad Method) - Agile development workflows
  › [x] BMB (BMad Builder) - Create custom agents
  › [ ] Game Development Features (not needed)

? Project name?
  › circuit-explainer

? Primary programming language?
  › Python

? IDE integration?
  › VSCode (or your preferred IDE)
```

### Step 3: Verify Installation

```bash
# Check installed structure
ls bmad/

# Expected output:
# core/  bmm/  bmb/  cis/  _cfg/

# View BMAD status
node tools/cli/bmad-cli.js status
# Or if npm scripts are set up:
npm run bmad:status
```

---

## Post-Installation: Understanding the Structure

### BMM Module (BMad Method)

**Location**: `bmad/bmm/`

**Purpose**: Core agile development framework with pre-built agents and workflows

**Key Directories**:
- `agents/` - Pre-built agents (architect, developer, tester, etc.)
- `workflows/` - Standard workflows (feature-development, refactoring, bugfix)
- `config.yaml` - Module configuration

### BMB Module (BMad Builder)

**Location**: `bmad/bmb/`

**Purpose**: Tools to create custom agents, workflows, and modules

**Key Workflows**:
- `create-agent/` - Build custom agents
- `create-workflow/` - Build custom workflows
- `create-module/` - Build complete modules
- `edit-module/` - Modify existing modules

### CIS Module (Creative Intelligence Suite)

**Location**: `bmad/cis/`

**Purpose**: Shared capabilities across modules

### Your Customizations

**Location**: `bmad/_cfg/`

**Purpose**: Your project-specific customizations
- Custom agent configurations
- Project-specific workflows
- Environment settings

---

## Using BMAD After Installation

### Option 1: Direct Node.js Execution

```bash
# Main CLI tool location: tools/cli/bmad-cli.js

# Check status
node tools/cli/bmad-cli.js status

# Other commands
node tools/cli/bmad-cli.js install
node tools/cli/bmad-cli.js [command]
```

### Option 2: NPM Scripts (if configured)

```bash
# Status
npm run bmad:status

# Installation
npm run bmad:install
```

### Option 3: Using Workflows Directly

Work with AI agents using the installed workflows:

```bash
# Example: Using the circuit-explainer project with BMAD workflows
# You would invoke agents through conversation with your AI assistant,
# referencing the workflows in bmad/bmm/workflows/
```

---

## Customizing for Circuit-Explainer

### Create Custom Agents for Your Project

Using BMB's create-agent workflow:

1. **Create Architect Agent for Circuits**
   ```
   Location: bmad/_cfg/agents/circuit-architect.xml
   Purpose: Design new circuit types, plan visualizations
   ```

2. **Create Implementer Agent for Streamlit**
   ```
   Location: bmad/_cfg/agents/streamlit-implementer.xml
   Purpose: Implement Streamlit UI components
   ```

3. **Create Testing Agent for Visualizations**
   ```
   Location: bmad/_cfg/agents/viz-tester.xml
   Purpose: Test attention heatmaps, circuit simulations
   ```

### Create Custom Workflows

Using BMB's create-workflow workflow:

1. **Add New Circuit Workflow**
   ```
   Location: bmad/_cfg/workflows/add-circuit/
   Purpose: Standard process for adding new circuit types
   ```

2. **Refactor Visualization Workflow**
   ```
   Location: bmad/_cfg/workflows/refactor-viz/
   Purpose: Extract and modularize visualization code
   ```

---

## Module Structure Reference

### Standard BMAD Module Layout

```
module-name/
├── agents/              # Agent definitions (XML format)
│   ├── agent-1.xml
│   └── agent-2.xml
├── workflows/           # Workflow definitions
│   ├── workflow-1/
│   │   ├── workflow.yaml
│   │   ├── instructions.md
│   │   └── templates/
│   └── workflow-2/
├── config.yaml          # Module configuration
└── README.md            # Module documentation
```

### Agent XML Format

```xml
<!-- Example agent structure -->
<agent>
  <name>Agent Name</name>
  <persona>Agent role and behavior</persona>
  <commands>
    <command>Specific instructions</command>
  </commands>
</agent>
```

### Workflow Structure

```
workflow-name/
├── workflow.yaml        # Workflow metadata and configuration
├── instructions.md      # Detailed workflow instructions
├── checklist.md        # Validation checklist (optional)
└── templates/          # Template files (optional)
```

---

## Troubleshooting Installation

### Installation Fails

**Check Node.js version**:
```bash
node --version
# Should be >= 20.0.0
```

**Verify npm permissions**:
```bash
npm config get prefix
# Ensure you have write permissions
```

**Clear caches**:
```bash
# Clear npm cache
npm cache clean --force

# Clear npx cache
npx clear-npx-cache
```

**Try installation again**:
```bash
npx bmad-method@alpha install
```

---

## Next Steps After Installation

### 1. Explore Installed Modules

```bash
# View BMM agents
ls bmad/bmm/agents/

# View BMM workflows
ls bmad/bmm/workflows/

# View BMB creation tools
ls bmad/bmb/workflows/
```

### 2. Review Documentation

```bash
# Read BMM module README
cat bmad/bmm/README.md

# Read BMB module README
cat bmad/bmb/README.md
```

### 3. Start Using BMAD

**Option A**: Use existing BMM workflows for software development

**Option B**: Create custom agents/workflows with BMB for circuit-explainer specific needs

**Option C**: Hybrid approach - Use BMM workflows, customize with BMB as needed

---

## Recommended Approach for Circuit-Explainer

### Phase 1: Use BMM Standard Workflows

Start with BMM's pre-built agile workflows:
- `feature-development` - For adding new circuits
- `refactoring` - For modularizing code
- `bugfix` - For fixing issues

### Phase 2: Customize with BMB

Create circuit-explainer specific agents:
- Circuit architect
- Streamlit implementer
- Visualization tester

Create custom workflows:
- Add new circuit type
- Extract visualization component
- Integrate TransformerLens

### Phase 3: Build Custom Module (Optional)

If the project grows significantly, create a complete custom module:
- Circuit-explainer specific agents
- Educational content workflows
- Research analysis workflows

---

## Integration with Your Development Workflow

### Working with AI Agents

Once BMAD is installed, you work with AI agents by:

1. **Referencing workflows** from `bmad/bmm/workflows/`
2. **Invoking agents** defined in `bmad/bmm/agents/`
3. **Following instructions** in workflow `instructions.md` files
4. **Using templates** from workflow `templates/` directories

### Example: Starting a Refactoring Task

```
You: "Let's use the BMM refactoring workflow to extract 
     visualization components from induction_heads_explorer.py"

AI Agent: 
- Reads bmad/bmm/workflows/refactoring/workflow.yaml
- Follows bmad/bmm/workflows/refactoring/instructions.md
- Acts as architect agent to plan refactoring
- Implements changes following checklist
- Validates no regressions
```

---

## Summary

### What You Get from BMAD Installation

✅ **Unified directory structure** - `bmad/` with all modules  
✅ **Pre-built agents** - Ready-to-use software development agents  
✅ **Standard workflows** - Feature dev, refactoring, bugfix processes  
✅ **Builder tools** - Create custom agents/workflows with BMB  
✅ **Configuration system** - Customizable via `_cfg/`  
✅ **IDE integration** - Works with your development environment  

### Installation Command

```bash
npx bmad-method@alpha install
```

This creates the foundation for systematic, agent-guided development of your circuit-explainer project using the BMAD-METHOD agile framework.

---

## References

- **Official Repository**: https://github.com/bmadcode/bmad-method
- **Documentation**: Context7 BMAD-METHOD docs
- **Version**: v6.0.0-alpha.0
- **Module Codes**: BMM (method), BMB (builder), CIS (creative suite)