from teenyagent.agent import Agent, CodeModel, Toolbox

toolbox = Toolbox()
code_model = CodeModel()

agent = Agent(code_model, toolbox)
result = agent.run("What is the capital of France?")
print(result)