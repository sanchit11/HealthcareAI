import asyncio
from dotenv import load_dotenv

from agents import Agent, Runner
import aiprompts.soapprompts as soapprompts

# Import your cleanly isolated schemas here
from ioschemas.SoapSchemas import SoapGenerationPayload, SoapNoteOutput

load_dotenv(override=True)


async def createAgent() -> Agent:
    """
    Initializes the agent with permanent system rules, model configurations,
    and output schema expectations.
    """
    return Agent(
        name="Professional Clinical Scribe Agent",
        instructions=soapprompts.get_system_prompt(),
        model="gpt-4o-mini",
        output_type=SoapNoteOutput
    )

async def generate_soap_note(payload: SoapGenerationPayload) -> SoapNoteOutput:
    """
    Generate SOAP note using OpenAI Agents SDK with structured inputs and outputs.
    """
    # Convert incoming Pydantic payload object back into a Python dict safely
    input_dict = payload.model_dump()

    # 1. Generate the dynamic user prompt containing the payload data
    user_prompt = soapprompts.generate_user_prompt(input_dict)

    # 2. Instantiate the agent with its system rules configuration
    agent = await createAgent()

    # 3. Execute the agent through the framework runner, feeding it the user prompt.
    #    Runner.run() returns a RunResult — the actual SoapNoteOutput is at .final_output
    run_result = await Runner.run(agent, user_prompt)

    return run_result.final_output