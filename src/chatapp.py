'''
Base classes for config
'''
import os

def iterset_envvars(obj,
                    envvars,
                    set_optional=False):
    '''
    Iterate a list of envvars
    '''
    for evar in envvars:
        if evar not in os.environ and not set_optional:
            raise ValueError(f'Required environment variable {evar} was not set')
        setattr(obj, evar.lower(), os.environ.get(evar))

class Chatapp():
    '''
    Chatapp class
    '''
    def __init__(self):

        self.vector = self.Vector()
        self.config = self.Config()
        self.model_type = None
        self.temperature = None
        self.system_prompt = None

        self.check_envvars()

        if self.vector.embed_type == 'openai' and not self.model_type == 'openai':
            self.openai = self.OpenAI()
        if self.model_type == 'openai':
            print("Found openai")
            self.openai = self.OpenAI(llm=True)

    def check_envvars(self):
        '''
        Check for required envvars
        '''
        base_config = [
                'MODEL_TYPE',
                'TEMPERATURE',
                'SYSTEM_PROMPT']
        self.iterset_envvars(base_config)

    def iterset_envvars(self,
                        envvars,
                        set_optional=False):
        '''
        Iterate a list of envvars
        '''
        for evar in envvars:
            if evar not in os.environ and not set_optional:
                raise ValueError(f'Required environment variable {evar} was not set')
            setattr(self, evar.lower(), os.environ.get(evar))

    # pylint: disable='too-few-public-methods'
    class OpenAI():
        '''
        OpenAI class
        '''
        def __init__(self, llm=False):
            self.openai_api_key = None
            self.openai_model_name = None

            iterset_envvars(self, ['OPENAI_API_KEY'])
            if llm:
                iterset_envvars(self, ['OPENAI_MODEL_NAME'])

    class Vector():
        '''
        Vector class
        '''
        def __init__(self):
            self.embed_type = None
            self.collection_name = None
            self.database_url = None
            self.vector_type = None

            self.set_vector_vars()
            self.heroku_fixup()

        def set_vector_vars(self):
            '''
            Set vector related vars
            '''
            default_vars = [
                'EMBED_TYPE',
                'VECTOR_TYPE',
                'COLLECTION_NAME',
                'TEMPERATURE',
            ]
            iterset_envvars(self, default_vars)
            if self.vector_type == 'pinecone':
                iterset_envvars(self, ['PINECONE_API_KEY'])
            if self.vector_type == 'pgvector':
                iterset_envvars(self, ['DATABASE_URL'])

        def heroku_fixup(self):
            '''
            Heroku fix up
            '''
            if self.database_url.startswith('postgres://'):
                self.database_url = self.database_url.replace('postgres://', 'postgresql://')

    class Config():
        '''
        Generic config class
        '''
        def __init__(self):
            self.enable_rephrase = True
            self.langchain_debug = False
            self.slack_debug = False
            self.welcome_message = False

            self.set_defaults()
            self.configure_welcome()

        def configure_welcome(self):
            '''
            Configure welome message
            '''
            if self.welcome_message:
                iterset_envvars(self, ['WELCOME_PROMPT'])
            else:
                os.environ['WELCOME_MESSAGE'] = False

        def set_defaults(self):
            '''
            Set defaults
            '''
            optional_config = [
                'ENABLE_REPHRASE',
                'LANGCHAIN_DEBUG',
                'SLACK_DEBUG',
                'WELCOME_MESSAGE'
                ]
            iterset_envvars(self,
                            optional_config,
                            set_optional=True)
