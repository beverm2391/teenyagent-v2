# Project Memory

## Project State
Core agent system with minimalist design philosophy. Error system and debug utilities implemented with rich console output and debug levels. Message system implemented with role-based formatting and cleaning. Tool system implemented with base class, decorator registration, and validation. Prompts system enhanced with structured code, tool calling, and single-step execution patterns. Task analysis and prompt selection implemented in HybridModel. Mock models implemented for testing.

## Goals
- Create lightweight agent with basic tools
- Custom tool registration
- Basic memory
- Model integration

## Recent Changes
- Fixed end-to-end test issues:
  - Added mock models for testing
  - Fixed final_answer handling in code execution
  - Added detailed debug output
  - First test passing (simple_calculation)
- Previous: Added task analysis and prompt selection to HybridModel
- Previous: Enhanced prompts system with three specialized prompts
- Previous: Completed project review and status assessment

## Design Limitations (Intentional)
- Sequential execution only - simpler, more predictable
- Basic memory without persistence - reduces complexity
- Limited tool history in context - forces concise interactions
- No streaming responses - cleaner implementation
- Manual tool registration - explicit over implicit
- No facts tracking or planning system - focused on core functionality
- Simple task analysis - keyword based rather than semantic
- Mock models for testing - reduces API dependencies

## Security Considerations
- Local Python execution needs sandboxing
- Tool permissions need scoping