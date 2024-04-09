# pragma pylint: disable=too-few-public-methods
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
        if evar in os.environ:
            setattr(obj, evar.lower(), os.environ.get(evar))

class Config():
    '''
    Generic config class
    '''
    def __init__(self):
        self.slack = self.SlackConfig()
        self.enable_rephrase = 'true'
        self.langchain_debug = 'false'
        self.welcome_message = 'false'
        self.welcome_purpose = None
        self.enable_sources = 'true'

        self._set_defaults()
        self._configure_welcome()

    def _configure_welcome(self):
        '''
        Configure welome message
        '''
        if self.welcome_message == 'true':
            iterset_envvars(self, ['WELCOME_PURPOSE'])

    def _set_defaults(self):
        '''
        Set defaults
        '''
        optional_config = [
            'ENABLE_REPHRASE',
            'LANGCHAIN_DEBUG',
            'WELCOME_MESSAGE',
            'ENABLE_SOURCES'
        ]
        iterset_envvars(self,
                        optional_config,
                        set_optional=True)

    class SlackConfig():
        '''
        Slack config
        '''
        def __init__(self):
            self.slack_debug = 'false'
            self.slack_bot_token = None
            self.slack_app_token = None
            self.slack_signing_secret = None
            self.enable_feedback = 'true'

            self._set_defaults()

        def _set_defaults(self):
            '''
            Set defaults
            '''
            base_config = [
                    'SLACK_BOT_TOKEN',
                    'SLACK_APP_TOKEN',
                    'SLACK_SIGNING_SECRET'
            ]
            optional_config = [
                    'SLACK_DEBUG',
                    'ENABLE_FEEDBACK'
            ]
            iterset_envvars(self,
                            base_config)
            iterset_envvars(self,
                            optional_config,
                            set_optional=True)

class Chatapp():
    '''
    Chatapp class
    '''
    def __init__(self, config):

        self.vector = self.Vector()
        self.config = config
        self.model_type = None
        self.temperature = None
        self.bot_purpose = None
        self.bot_support = None
        self._check_envvars()

        if self.vector.embed_type == 'openai' and not self.model_type == 'openai':
            self.openai = self.OpenAI()
        if self.model_type == 'openai':
            print("Found openai")
            self.openai = self.OpenAI(llm=True)

    def _check_envvars(self):
        '''
        Check for required envvars
        '''
        base_config = [
                'MODEL_TYPE',
                'TEMPERATURE',
                'BOT_PURPOSE',
                'BOT_SUPPORT']
        iterset_envvars(self, base_config)

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

            self._set_vector_vars()
            self._heroku_fixup()

        def _set_vector_vars(self):
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

        def _heroku_fixup(self):
            '''
            Heroku fix up
            '''
            if self.database_url.startswith('postgres://'):
                self.database_url = self.database_url.replace('postgres://', 'postgresql://')
