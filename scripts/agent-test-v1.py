from teenyagent.agent import Agent, CodeModel, Toolbox

toolbox = Toolbox()
code_model = CodeModel()

agent = Agent(code_model, toolbox)
result = agent.run("Who is Ben Everman?")