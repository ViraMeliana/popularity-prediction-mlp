import connexion
from flask_cors import CORS


core = connexion.App(__name__, specification_dir='./routes')
CORS(core.app)
core.add_api('routes.yml')
