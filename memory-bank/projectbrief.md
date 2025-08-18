# Project Brief: Saga Knowledge Graph System

## Overview
The Saga system is a generative narrative framework that creates structured stories using an agent-based architecture with knowledge graph integration. The system generates chapters through a coordinated workflow of specialized agents that maintain narrative consistency and enrich world elements.

## Core Architecture
- **Agent-Based Workflow**: 
  - NarrativeAgent: Handles scene planning and chapter generation
  - RevisionAgent: Performs continuity and quality validation
  - KnowledgeAgent: Manages knowledge graph updates and enrichment

- **Knowledge Graph Integration**:
  - Neo4j database for persistent world state
  - Dynamic updating of character, world, and plot elements
  - Automatic consistency checking across narrative elements

## Key Features
1. **Consolidated Agent Architecture**: 
   - Reduced module count through strategic consolidation
   - Improved coordination between agent roles
   - Enhanced maintainability and testability

2. **Structured Output Management**:
   - All persistent outputs stored in `novel_output/` directory
   - Dedicated log files with structured naming
   - Consistent file organization for easy tracking

3. **Enhanced Data Tracking**:
   - Comprehensive usage data aggregation from planning to drafting
   - Detailed metrics for each chapter generation cycle
   - Real-time monitoring of system performance

## Goals
- Create cohesive, consistent narratives across multiple chapters
- Maintain accurate world state through knowledge graph updates
- Provide transparent tracking of the narrative generation process
- Enable easy debugging and auditing of the system's decisions
