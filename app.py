"""
Data Architect Agent - Flask Backend
Windows-compatible web server with Natural Language Query capability
Version 15 - Bug Fix: Upload Controls Display Issue
"""

from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import os
import json
import traceback
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Debug: Print the value of AZURE_OPENAI_API_KEY to verify loading
print("AZURE_OPENAI_API_KEY:", os.getenv('AZURE_OPENAI_API_KEY'))

# Import core agents
from core_agents import (
    DataArchitectAPI,
    TemperatureAdjustedLLM,
    AGENT_TEMPERATURES
)

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
app.secret_key = os.urandom(24)  # For session management
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
try:
    llm_provider = os.getenv('LLM_PROVIDER', 'azure').lower()
    
    if llm_provider == 'azure':
        # Azure OpenAI Configuration (Shell Enterprise)
        # Custom approach: Create httpx client manually to avoid proxies parameter
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4')
        azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        
        if not azure_endpoint or not azure_api_key or azure_api_key == 'your-azure-api-key-here':
            logger.error("❌ Azure OpenAI not configured in .env file")
            logger.error("Required: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY")
            llm = None
        else:
            # Import httpx and create client manually
            import httpx
            from openai import AzureOpenAI
            
            # We'll create fresh clients per request, so no global client needed
            try:
                
                # Create a simple wrapper that mimics LangChain's interface
                class SimpleAzureLLM:
                    def __init__(self, azure_endpoint, api_key, api_version, deployment):
                        # Store config instead of client - we'll recreate client per request
                        self.azure_endpoint = azure_endpoint
                        self.api_key = api_key
                        self.api_version = api_version
                        self.deployment = deployment
                        self.model_name = deployment
                    
                    def _create_fresh_client(self):
                        """Create a fresh Azure OpenAI client with new http connection"""
                        import httpx
                        from openai import AzureOpenAI
                        
                        # Better timeout/retry configuration
                        http_client = httpx.Client(
                            timeout=httpx.Timeout(
                                connect=10.0,  # Connection timeout
                                read=60.0,     # Read timeout
                                write=10.0,    # Write timeout
                                pool=5.0       # Pool timeout
                            ),
                            follow_redirects=True,
                            limits=httpx.Limits(
                                max_connections=100,
                                max_keepalive_connections=20,
                                keepalive_expiry=30.0  # Close idle connections after 30s
                            )
                        )
                        
                        # Create Azure OpenAI client
                        return AzureOpenAI(
                            azure_endpoint=self.azure_endpoint,
                            api_key=self.api_key,
                            api_version=self.api_version,
                            http_client=http_client,
                            max_retries=3
                        )
                    
                    def invoke(self, prompt):
                        """Simple invoke method compatible with our agents"""
                        if isinstance(prompt, str):
                            messages = [{"role": "user", "content": prompt}]
                        else:
                            messages = prompt
                        
                        # Recreate client for each request (fresh connection)
                        client = self._create_fresh_client()
                        
                        try:
                            response = client.chat.completions.create(
                                model=self.deployment,
                                messages=messages,
                                temperature=0.1
                            )
                            
                            # Return in LangChain-compatible format
                            class Response:
                                def __init__(self, content):
                                    self.content = content
                            
                            return Response(response.choices[0].message.content)
                        except Exception as e:
                            # Enhanced error handling for deployment errors
                            error_msg = str(e)
                            if '404' in error_msg and 'DeploymentNotFound' in error_msg:
                                raise Exception(
                                    f"Deployment '{self.deployment}' not found at {self.azure_endpoint}. "
                                    f"Please check your AZURE_OPENAI_DEPLOYMENT_NAME in .env file. "
                                    f"Common deployment names: gpt-4o, gpt-4-turbo, gpt-35-turbo"
                                )
                            raise
                        finally:
                            # Close the http client to free resources
                            if hasattr(client, '_client') and hasattr(client._client, 'close'):
                                try:
                                    client._client.close()
                                except:
                                    pass
                
                # Create wrapper with config (not client instance)
                base_llm = SimpleAzureLLM(azure_endpoint, azure_api_key, azure_api_version, azure_deployment)
                
                # V12 NEW: Validate deployment on startup with a test call
                logger.info("🔍 Validating Azure OpenAI deployment...")
                try:
                    test_response = base_llm.invoke("Hello")
                    logger.info(f"✅ Azure OpenAI deployment validated: {azure_deployment}")
                except Exception as e:
                    error_msg = str(e)
                    if '404' in error_msg or 'DeploymentNotFound' in error_msg:
                        logger.error("=" * 70)
                        logger.error("❌ DEPLOYMENT NOT FOUND ERROR")
                        logger.error("=" * 70)
                        logger.error(f"Deployment Name: {azure_deployment}")
                        logger.error(f"Endpoint: {azure_endpoint}")
                        logger.error("")
                        logger.error("SOLUTION:")
                        logger.error("1. Check Azure Portal → Your OpenAI Resource → Deployments")
                        logger.error("2. Find the actual deployment name (e.g., 'gpt-4o', 'gpt-4-turbo')")
                        logger.error("3. Update AZURE_OPENAI_DEPLOYMENT_NAME in .env file")
                        logger.error("")
                        logger.error("Common deployment names:")
                        logger.error("  - gpt-4o (GPT-4 Omni - newest)")
                        logger.error("  - gpt-4-turbo")
                        logger.error("  - gpt-4o-mini")
                        logger.error("  - gpt-35-turbo")
                        logger.error("=" * 70)
                        # Don't fail completely - allow app to start but warn
                        logger.warning("⚠️  App will start but queries will fail until deployment is fixed")
                    else:
                        logger.error(f"❌ Unexpected error during validation: {e}")
                        raise
                
                llm = TemperatureAdjustedLLM(base_llm, AGENT_TEMPERATURES)
                logger.info(f"✅ Azure OpenAI initialized with deployment: {azure_deployment}")
                logger.info(f"✅ Endpoint: {azure_endpoint}")
                logger.info(f"✅ Connection config: 3 retries, 30s keepalive, fresh client per request")
            except Exception as e:
                logger.error(f"❌ Azure OpenAI initialization failed: {e}")
                logger.error(f"Full error: {traceback.format_exc()}")
                llm = None
    
    else:
        # Standard OpenAI Configuration
        from langchain_openai import ChatOpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your-api-key-here':
            logger.error("OpenAI API key not configured in .env file")
            llm = None
        else:
            model = os.getenv('OPENAI_MODEL', 'gpt-4')
            base_llm = ChatOpenAI(model=model, temperature=0.1)
            llm = TemperatureAdjustedLLM(base_llm, AGENT_TEMPERATURES)
            logger.info(f"LLM initialized with model: {model}")
            
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    logger.error(f"Full error: {traceback.format_exc()}")
    llm = None

# Initialize API
base_path = os.getenv('BASE_PATH', './projects')
if llm:
    api = DataArchitectAPI(llm, base_path=base_path)
    logger.info(f"Data Architect API initialized with base_path: {base_path}")
else:
    api = None
    logger.warning("API not initialized - configure Azure OpenAI in .env file")

# Session storage (in production, use Redis or database)
sessions = {}

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'llm_configured': llm is not None,
        'api_ready': api is not None
    })

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Initialize a new conversation session"""
    try:
        session_id = f"session_{len(sessions) + 1}"
        sessions[session_id] = {
            'stage': 0,
            'context': {},
            'history': [],
            'query_data': {}  # Store data for querying
        }
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Session started'
        })
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process_message', methods=['POST'])
def process_message():
    """Process user message and return response"""
    try:
        if not api:
            return jsonify({
                'success': False,
                'error': 'API not configured. Please check Azure OpenAI settings in .env file'
            }), 500
        
        data = request.json
        session_id = data.get('session_id')
        user_input = data.get('message', '').strip()
        
        if not session_id or session_id not in sessions:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        if not user_input:
            return jsonify({'success': False, 'error': 'Empty message'}), 400
        
        session_data = sessions[session_id]
        stage = session_data['stage']
        context = session_data['context']
        history = session_data['history']
        
        # Add user message to history
        history.append({'role': 'user', 'content': user_input})
        
        # Stage-based conversation flow
        response_data = {}
        
        # Stage 0: Get domain
        if stage == 0:
            context['domain'] = user_input
            response_data = {
                'success': True,
                'message': f"Great! You've chosen the {user_input} domain.\n\n"
                          f"Now, choose your approach:\n"
                          f"• Type 'forward' for Forward Engineering (requirements → models)\n"
                          f"• Type 'reverse' for Reverse Engineering (DDL/CSV → models)",
                'stage': 1,
                'context': context
            }
            session_data['stage'] = 1
        
        # Stage 1: Get mode
        elif stage == 1:
            mode = user_input.lower()
            if mode in ['forward', 'reverse']:
                context['mode'] = mode
                
                if mode == 'forward':
                    response_data = {
                        'success': True,
                        'message': "Perfect! Let's do Forward Engineering.\n\n"
                                  "Please describe your system requirements in detail. "
                                  "What kind of data system do you need to build?",
                        'stage': 2,
                        'context': context
                    }
                    session_data['stage'] = 2
                else:  # reverse
                    response_data = {
                        'success': True,
                        'message': "Perfect! Let's do Reverse Engineering.\n\n"
                                  "Choose your input source:\n"
                                  "• Type 'DDL' to upload SQL schema files\n"
                                  "• Type 'CSV' to upload data files",
                        'stage': 3,
                        'context': context
                    }
                    session_data['stage'] = 3
            else:
                response_data = {
                    'success': True,
                    'message': "Please type either 'forward' or 'reverse'",
                    'stage': 1,
                    'context': context
                }
        
        # Stage 2: Get requirements (forward engineering)
        elif stage == 2:
            context['requirements'] = user_input
            response_data = {
                'success': True,
                'message': "Excellent! I understand your requirements.\n\n"
                          "Now, please list the main entities (tables) you want in your system.\n"
                          "For example: Customer, Order, Product, Payment\n\n"
                          "List them separated by commas:",
                'stage': 4,
                'context': context
            }
            session_data['stage'] = 4
        
        # Stage 3: Get input type (reverse engineering)
        elif stage == 3:
            input_type = user_input.upper()
            if input_type in ['DDL', 'CSV']:
                context['input_type'] = input_type
                response_data = {
                    'success': True,
                    'message': f"Great! Please upload your {input_type} file(s).",
                    'stage': 5,
                    'show_upload': True,
                    'context': context
                }
                session_data['stage'] = 5
            else:
                response_data = {
                    'success': True,
                    'message': "Please type either 'DDL' or 'CSV'",
                    'stage': 3,
                    'context': context
                }
        
        # Stage 4: Get entities list (forward engineering)
        elif stage == 4:
            context['entities'] = [e.strip() for e in user_input.split(',')]
            
            # Start forward engineering process
            response_data = {
                'success': True,
                'message': "Perfect! Starting forward engineering process...\n"
                          f"Domain: {context['domain']}\n"
                          f"Entities: {', '.join(context['entities'])}\n\n"
                          "This will take a few minutes. Please wait...",
                'stage': 6,
                'start_processing': True,
                'context': context
            }
            session_data['stage'] = 6
        
        # Add response to history
        if 'message' in response_data:
            history.append({'role': 'assistant', 'content': response_data['message']})
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}\\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Error: {str(e)}'
        }), 500

@app.route('/api/generate', methods=['POST'])
def generate_architecture():
    """Generate architecture based on requirements"""
    try:
        if not api:
            return jsonify({
                'success': False,
                'error': 'API not configured'
            }), 500
        
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id or session_id not in sessions:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        session_data = sessions[session_id]
        context = session_data['context']
        
        # Generate architecture
        domain = context.get('domain', 'Unknown')
        requirements = context.get('requirements', '')
        entities = context.get('entities', [])
        
        # Create project
        project_id = f"{domain.lower().replace(' ', '_')}_forward"
        
        # FIXED: Use correct method name create_project with explicit_entities
        project_request = {
            "project_id": project_id,
            "description": f"Domain: {domain}",
            "requirements": requirements,
            "explicit_entities": entities,  # Pass entities explicitly
            "create_zip": True
        }
        
        result = api.create_project(project_request)
        
        # FIXED: Get database path from result or search for it (like DDL does)
        db_path = result.get('database_path')
        
        if not db_path:
            # Try multiple possible locations
            base_path = Path('./projects')
            possible_paths = [
                # Pattern 1: Root projects folder with _db.db suffix
                base_path / f'{project_id}_db.db',
                # Pattern 2: Subdirectory with standard name
                base_path / project_id / '06_database_provisioning' / f'{project_id}.db',
                # Pattern 3: Root projects folder with standard name
                base_path / f'{project_id}.db',
            ]
            
            for path in possible_paths:
                if path.exists():
                    db_path = path
                    logger.info(f"✅ Found database at: {db_path}")
                    break
            
            if not db_path or not Path(db_path).exists():
                db_path = possible_paths[1]
                logger.warning(f"⚠️  Database not found at any expected location")
        
        # FIXED: Store query data for natural language query feature (like DDL does)
        if db_path:
            # Get DDL content from result - try multiple possible locations
            ddl_content = result.get('physical_model', '')
            
            # If not in top level, check in state or artifacts
            if not ddl_content:
                # Try to get from workflow state
                if hasattr(api, 'workflow') and hasattr(api.workflow, 'state'):
                    ddl_content = api.workflow.state.get('physical_model', '')
                    logger.info(f"   Found DDL in workflow state: {len(ddl_content)} chars")
            
            # If still not found, try to read from file
            if not ddl_content and result.get('artifacts'):
                physical_ddl_path = result['artifacts'].get('physical_ddl')
                logger.info(f"   Raw artifact path: {repr(physical_ddl_path)}")
                
                if physical_ddl_path:
                    # Convert path string to Path object
                    full_ddl_path = Path(str(physical_ddl_path))
                    logger.info(f"   Attempting to read DDL from: {full_ddl_path}")
                    logger.info(f"   Path exists: {full_ddl_path.exists()}")
                    
                    if full_ddl_path.exists():
                        try:
                            with open(full_ddl_path, 'r', encoding='utf-8') as f:
                                ddl_content = f.read()
                            logger.info(f"   ✅ Read DDL from file ({len(ddl_content)} chars)")
                        except Exception as e:
                            logger.error(f"   ❌ Failed to read DDL file: {e}")
                    else:
                        logger.warning(f"   ⚠️  DDL file not found at: {full_ddl_path}")
            
            # Last resort: Generate DDL from database schema
            if not ddl_content and Path(db_path).exists():
                logger.info(f"   Attempting to extract DDL from database schema...")
                try:
                    import sqlite3
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # Get all table creation statements
                    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
                    ddl_statements = [row[0] for row in cursor.fetchall()]
                    ddl_content = '\n\n'.join(ddl_statements)
                    conn.close()
                    
                    logger.info(f"   ✅ Extracted DDL from database ({len(ddl_content)} chars)")
                except Exception as e:
                    logger.error(f"   ❌ Failed to extract DDL from database: {e}")
            
            session_data['query_data'] = {
                'ddl': ddl_content,
                'db_path': str(db_path),
                'project_id': project_id
            }
            
            logger.info(f"✅ Stored query data for forward engineering session {session_id}")
            logger.info(f"   Database path: {db_path}")
            logger.info(f"   Database exists: {Path(db_path).exists()}")
            logger.info(f"   DDL length: {len(ddl_content)} characters")
            
            if not ddl_content:
                logger.warning(f"   ⚠️  WARNING: DDL content is empty! Queries may fail.")
        
        # FIXED: Return full result structure like reverse engineering endpoints
        response_data = {
            'success': True,
            'project_id': project_id,
            'message': 'Architecture generated successfully!',
            'result': {
                'project_id': project_id,
                'status': result.get('status'),
                'current_step': result.get('current_step'),
            }
        }
        
        # Add artifacts if available
        if result.get('artifacts'):
            # FIXED: Normalize paths for web viewing
            artifacts = result['artifacts']
            normalized_artifacts = {}
            
            for key, path in artifacts.items():
                # Convert to string and normalize
                path_str = str(path)
                
                # Make path relative to projects directory
                if './projects/' in path_str or '.\\projects\\' in path_str:
                    # Remove the ./projects/ or .\projects\ prefix
                    path_str = path_str.split('projects', 1)[1].lstrip('/\\')
                elif 'projects' in path_str:
                    # Handle cases where path contains projects directory
                    parts = path_str.split('projects')
                    if len(parts) > 1:
                        path_str = parts[1].lstrip('/\\')
                
                # Convert backslashes to forward slashes for web
                path_str = path_str.replace('\\', '/')
                
                normalized_artifacts[key] = path_str
                logger.info(f"   Normalized artifact {key}: {path_str}")
            
            response_data['result']['artifacts'] = normalized_artifacts
        
        # Add zip file if available
        if result.get('zip_file'):
            response_data['result']['zip_file'] = result['zip_file']
        
        # FIXED: Add database path (like DDL does)
        if db_path:
            response_data['result']['database_path'] = str(db_path)
        
        # FIXED: Enable query button if database exists (like DDL does)
        if db_path and Path(db_path).exists():
            response_data['enable_query'] = True
            logger.info(f"✅ Query feature enabled for forward engineering")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error generating architecture: {e}\\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Generation error: {str(e)}'
        }), 500

@app.route('/api/upload_ddl', methods=['POST'])
def upload_ddl():
    """Handle DDL file upload for reverse engineering"""
    try:
        if not api:
            return jsonify({
                'success': False,
                'error': 'API not configured'
            }), 500
        
        # Debug: Log what we received
        logger.info("🔍 DDL Upload Debug:")
        logger.info(f"   request.json: {request.json if request.is_json else 'Not JSON'}")
        
        # DDL comes as JSON (not FormData like CSV)
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
            
        session_id = data.get('session_id')
        ddl_content = data.get('ddl_content')
        
        logger.info(f"   Extracted session_id: {session_id}")
        logger.info(f"   DDL content length: {len(ddl_content) if ddl_content else 0}")
        logger.info(f"   Available sessions: {list(sessions.keys())}")
        
        if not session_id or session_id not in sessions:
            logger.error(f"   ❌ Invalid session: '{session_id}' not in {list(sessions.keys())}")
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        if not ddl_content:
            return jsonify({'success': False, 'error': 'No DDL content provided'}), 400
        
        session_data = sessions[session_id]
        context = session_data['context']
        domain = context.get('domain', 'Unknown')
        
        # Create project
        project_id = f"{domain.lower().replace(' ', '_')}_ddl_reverse"
        
        # Run reverse engineering using the correct method
        project_request = {
            "project_id": project_id,
            "ddl_content": ddl_content,
            "domain": domain,
            "create_zip": True
        }
        
        result = api.create_project_from_ddl(project_request)
        
        # Get database path from result or search for it
        db_path = result.get('database_path')
        
        if not db_path:
            # Try multiple possible locations
            base_path = Path('./projects')
            possible_paths = [
                # Pattern 1: Root projects folder with _db.db suffix
                base_path / f'{project_id}_db.db',
                # Pattern 2: Subdirectory with standard name
                base_path / project_id / '06_database_provisioning' / f'{project_id}.db',
                # Pattern 3: Root projects folder with standard name
                base_path / f'{project_id}.db',
            ]
            
            for path in possible_paths:
                if path.exists():
                    db_path = path
                    logger.info(f"✅ Found database at: {db_path}")
                    break
            
            if not db_path or not Path(db_path).exists():
                db_path = possible_paths[1]
                logger.warning(f"⚠️  Database not found at any expected location")
        
        # Store query data for natural language query feature
        session_data['query_data'] = {
            'ddl': ddl_content,
            'db_path': str(db_path),
            'project_id': project_id
        }
        
        logger.info(f"✅ Stored query data for DDL session {session_id}")
        logger.info(f"   Database path: {db_path}")
        logger.info(f"   Database exists: {Path(db_path).exists() if db_path else False}")
        
        # Build response with artifacts
        response_data = {
            'success': True,
            'message': 'DDL processed successfully!',
            'project_id': result.get('project_id'),
            'result': {
                'project_id': result.get('project_id'),
                'status': result.get('status'),
                'current_step': result.get('current_step'),
            }
        }
        
        # Add artifacts if available
        if result.get('artifacts'):
            response_data['result']['artifacts'] = result['artifacts']
        
        # Add database path
        if db_path:
            response_data['result']['database_path'] = str(db_path)
        
        # Enable query button if database exists
        if db_path and Path(db_path).exists():
            response_data['enable_query'] = True
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing DDL: {e}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Upload error: {str(e)}'
        }), 500

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload for reverse engineering"""
    try:
        if not api:
            return jsonify({
                'success': False,
                'error': 'API not configured'
            }), 500
        
        # Debug: Log what we received
        logger.info("🔍 CSV Upload Debug:")
        logger.info(f"   request.form: {dict(request.form)}")
        logger.info(f"   request.files: {request.files}")
        
        session_id = request.form.get('session_id')
        logger.info(f"   Extracted session_id: {session_id}")
        logger.info(f"   Available sessions: {list(sessions.keys())}")
        
        if not session_id or session_id not in sessions:
            logger.error(f"   ❌ Invalid session: '{session_id}' not in {list(sessions.keys())}")
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        # Handle multiple files
        files = request.files.getlist('files')
        if not files:
            return jsonify({'success': False, 'error': 'No files uploaded'}), 400
        
        # Save files temporarily
        import tempfile
        temp_dir = tempfile.mkdtemp()
        csv_files = []
        
        for file in files:
            if file.filename:
                file_path = os.path.join(temp_dir, file.filename)
                file.save(file_path)
                csv_files.append(file_path)
        
        session_data = sessions[session_id]
        context = session_data['context']
        domain = context.get('domain', 'Unknown')
        
        # Create project
        project_id = f"{domain.lower().replace(' ', '_')}_csv_reverse"
        
        # Run reverse engineering using the correct method from core_agents.py
        project_request = {
            "project_id": project_id,
            "description": f"Reverse engineering from CSV files for {domain}",
            "create_zip": True
        }
        
        result = api.create_project_from_csvs(
            project_request=project_request,
            csv_paths=csv_files,
            db_url=None
        )
        
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir)
        
        # Get database path from result or search for it
        db_path = result.get('database_path')
        
        if not db_path:
            # Try multiple possible locations where the database might be
            base_path = Path('./projects')
            possible_paths = [
                # Pattern 1: Root projects folder with _db.db suffix
                base_path / f'{project_id}_db.db',
                # Pattern 2: Subdirectory with _db.db suffix (DataInferenceAgent creates here)
                base_path / project_id / f'{project_id}_db.db',
                # Pattern 3: Subdirectory with standard name
                base_path / project_id / '06_database_provisioning' / f'{project_id}.db',
                # Pattern 4: Root projects folder with standard name
                base_path / f'{project_id}.db',
            ]
            
            for path in possible_paths:
                if path.exists():
                    db_path = path
                    logger.info(f"✅ Found database at: {db_path}")
                    break
            
            if not db_path or not Path(db_path).exists():
                db_path = possible_paths[1]
                logger.warning(f"⚠️  Database not found at any expected location")
        
        # Store query data for natural language query feature
        session_data['query_data'] = {
            'ddl': result.get('ddl_content', ''),  # Get DDL from result
            'db_path': str(db_path),
            'project_id': project_id
        }
        
        logger.info(f"✅ Stored query data for CSV session {session_id}")
        logger.info(f"   Database path: {db_path}")
        logger.info(f"   Database exists: {Path(db_path).exists() if db_path else False}")
        
        # Build response with artifacts
        response_data = {
            'success': True,
            'message': 'CSV files processed successfully!',
            'project_id': result.get('project_id'),
            'result': {
                'project_id': result.get('project_id'),
                'status': result.get('status'),
                'current_step': result.get('current_step'),
            }
        }
        
        # Add artifacts if available
        if result.get('artifacts'):
            response_data['result']['artifacts'] = result['artifacts']
        
        # Add database path
        if db_path:
            response_data['result']['database_path'] = str(db_path)
        
        # Enable query button if database exists
        if db_path and Path(db_path).exists():
            response_data['enable_query'] = True
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Upload error: {str(e)}'
        }), 500

@app.route('/api/query', methods=['POST'])
def query_database():
    """
    V12 ENHANCED: Natural language query endpoint with better error handling
    """
    try:
        # Get session data
        data = request.json
        session_id = data.get('session_id')
        nl_query = data.get('query', '').strip()
        
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid or expired session. Please refresh and try again.'}), 400
        
        if not nl_query:
            return jsonify({'error': 'Please enter a query'}), 400
        
        session_data = sessions[session_id]
        query_data = session_data.get('query_data', {})
        
        if not query_data:
            return jsonify({'error': 'No database available for querying. Please complete reverse engineering first.'}), 400
        
        ddl = query_data.get('ddl')
        db_path = query_data.get('db_path')
        
        if not ddl or not db_path:
            return jsonify({'error': 'Query session data incomplete. Please re-upload your DDL.'}), 400
        
        # V12 ENHANCED: Check if database file exists
        if not os.path.exists(db_path):
            return jsonify({
                'error': f'Database file not found. Please complete reverse engineering first.',
                'details': f'Expected path: {db_path}'
            }), 404
        
        # Generate SQL from natural language
        sql_query = generate_sql_from_nl(nl_query, ddl, llm)
        
        if not sql_query:
            return jsonify({'error': 'Failed to generate SQL query. Please try rephrasing your question.'}), 500
        
        # Execute the query
        results, error = execute_query(db_path, sql_query)
        
        if error:
            return jsonify({
                'error': f'Database error: {error}',
                'sql': sql_query,
                'help': 'The SQL was generated but failed to execute. Check the query syntax or try a different question.'
            }), 500
        
        return jsonify({
            'sql': sql_query,
            'results': results,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}\\n{traceback.format_exc()}")
        
        # V12 ENHANCED: Better error messages
        error_msg = str(e)
        if '404' in error_msg and 'DeploymentNotFound' in error_msg:
            return jsonify({
                'error': 'Azure OpenAI deployment not found',
                'details': 'The deployment name in your .env file does not exist. Check AZURE_OPENAI_DEPLOYMENT_NAME setting.',
                'help': 'Contact your IT administrator or check Azure Portal for the correct deployment name.'
            }), 500
        else:
            return jsonify({
                'error': f'Query processing error: {error_msg}',
                'help': 'Please try again or contact support if the problem persists.'
            }), 500

def generate_sql_from_nl(nl_query: str, ddl: str, llm_instance) -> str:
    """
    V12 ENHANCED: Convert natural language query to SQL with better error handling
    """
    try:
        prompt = f"""You are a SQL expert. Convert the following natural language query into a SQL query.

Database Schema (DDL):
{ddl}

Natural Language Query: {nl_query}

Instructions:
- Generate ONLY the SQL query, no explanations
- Use proper SQL syntax for SQLite
- Include appropriate JOINs if multiple tables are involved
- Use meaningful aliases
- Format the query for readability

SQL Query:"""

        # Use the llm instance directly (it's already configured with Azure OpenAI)
        response = llm_instance.base_llm.invoke(prompt)
        sql_query = response.content.strip()
        
        # Clean up the response (remove markdown code blocks if present)
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        
        logger.info(f"✅ Generated SQL: {sql_query}")
        return sql_query
        
    except Exception as e:
        error_msg = str(e)
        
        # V12 ENHANCED: Better logging for deployment errors
        if '404' in error_msg and 'DeploymentNotFound' in error_msg:
            logger.error("=" * 70)
            logger.error("❌ QUERY FAILED - DEPLOYMENT NOT FOUND")
            logger.error("=" * 70)
            logger.error(f"User Query: {nl_query}")
            logger.error(f"Error: {error_msg}")
            logger.error("")
            logger.error("ACTION REQUIRED:")
            logger.error("Update AZURE_OPENAI_DEPLOYMENT_NAME in .env file")
            logger.error("Common names: gpt-4o, gpt-4-turbo, gpt-35-turbo")
            logger.error("=" * 70)
        else:
            logger.error(f"Error generating SQL: {error_msg}")
            logger.error(f"Full trace: {traceback.format_exc()}")
        
        return None

def execute_query(db_path: str, sql_query: str):
    """
    Execute SQL query and return results
    Returns: (results, error)
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql_query)
            
            # For SELECT queries, fetch results
            if sql_query.strip().upper().startswith('SELECT'):
                rows = cursor.fetchall()
                # Convert to list of dictionaries
                results = [dict(row) for row in rows]
                logger.info(f"✅ Query executed successfully, returned {len(results)} rows")
                return results, None
            else:
                # For INSERT, UPDATE, DELETE
                conn.commit()
                return [{'affected_rows': cursor.rowcount}], None
                
        finally:
            conn.close()
            
    except sqlite3.Error as e:
        logger.error(f"❌ Database error: {e}")
        return None, str(e)
    except Exception as e:
        logger.error(f"❌ Query execution error: {e}")
        return None, str(e)

@app.route('/api/reset_session', methods=['POST'])
def reset_session():
    """Reset a session"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id and session_id in sessions:
            sessions[session_id] = {
                'stage': 0,
                'context': {},
                'history': [],
                'query_data': {}
            }
            return jsonify({'success': True, 'message': 'Session reset'})
        else:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
            
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Download generated files"""
    try:
        from pathlib import Path
        import os
        
        # Projects directory is the base_path
        projects_dir = Path(base_path)
        file_path = projects_dir / filename
        
        # Also check parent of projects_dir (for ZIP files at root)
        if not file_path.exists():
            file_path = projects_dir.parent / filename
        
        # Security check: ensure file is within allowed directories
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        allowed_dir1 = projects_dir.resolve()
        allowed_dir2 = projects_dir.parent.resolve()
        file_resolved = file_path.resolve()
        
        if not (str(file_resolved).startswith(str(allowed_dir1)) or 
                str(file_resolved).startswith(str(allowed_dir2))):
            logger.error(f"Security violation: {file_path}")
            return jsonify({'error': 'Access denied'}), 403
        
        # Send the file
        return send_from_directory(
            file_path.parent,
            file_path.name,
            as_attachment=True,
            download_name=file_path.name
        )
    except Exception as e:
        logger.error(f"Error downloading file: {e}\\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/view/<path:filepath>', methods=['GET'])
def view_file(filepath):
    """View HTML files in browser (for visualizations)"""
    try:
        from pathlib import Path
        
        # FIXED: Remove 'projects/' prefix from filepath if present to avoid duplication
        filepath = str(filepath)
        if filepath.startswith('projects/') or filepath.startswith('projects\\'):
            filepath = filepath.split('projects', 1)[1].lstrip('/\\')
        
        # Projects directory is the base_path  
        projects_dir = Path(base_path)
        file_path = projects_dir / filepath
        
        # Security check
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return "File not found", 404
            
        if not str(file_path.resolve()).startswith(str(projects_dir.resolve())):
            logger.error(f"Security violation: {file_path}")
            return "Access denied", 403
        
        # For HTML files, serve them for viewing (not download)
        if file_path.suffix == '.html':
            return send_from_directory(
                file_path.parent,
                file_path.name,
                mimetype='text/html'
            )
        else:
            # For other files, trigger download
            return send_from_directory(
                file_path.parent,
                file_path.name,
                as_attachment=True
            )
    except Exception as e:
        logger.error(f"Error viewing file: {e}\\n{traceback.format_exc()}")
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    # Create projects directory if it doesn't exist
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    # Print startup information
    print("\\n" + "="*70)
    print("Data Architect Agent - Flask Backend v16")
    print("="*70)
    print(f"LLM Status: {'✅ Configured' if llm else '❌ Not Configured'}")
    print(f"API Status: {'✅ Ready' if api else '❌ Not Ready'}")
    print(f"Projects Path: {Path(base_path).resolve()}")
    print(f"Server URL: http://localhost:5000")
    print(f"🔍 Natural Language Query Feature: {'✅ Enabled' if llm else '❌ Disabled'}")
    print("="*70)
    
    if not llm:
        print("\\n⚠️  WARNING: LLM not configured!")
        print("Please check your .env file and Azure OpenAI settings.")
        print("See VERSION_13_NOTES.md for troubleshooting.\\n")
    
    print("\\nPress Ctrl+C to stop the server\\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
