import os
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from datetime import datetime

from azureml.core import Experiment, RunConfiguration, Workspace
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.environment import DEFAULT_GPU_IMAGE
# from azureml.studio.common.utils.fileutil import path_from_project_root
from azureml.studio.tool.module_registry import ModuleRegistry, ServerConf, _init_ssl_context
from azureml.studio.tool.custom_module_reg import _register_custom_module


class DefaultConstants:
    WORKSPACE_NAME = 'ModuleX-NEU'
    SUBSCRIPTION_ID = 'e9b2ec51-5c94-4fa8-809a-dc1e695e4896'
    TENANT_ID = '72f988bf-86f1-41af-91ab-2d7cd011db47'
    RESOURCE_GROUP = 'ModuleX-rg'
    COMPUTE_NAME = 'gpu-nc6'
    CERT_FILE_NAME = 'cert.pem'
    GLOBAL_MODULE_ID = '506153734175476c4f62416c57734963'

    
class WorkspaceRegistry:
    """
    Workspace Registry class is used for sending requests to admin api to obtain workspace id

    """

    def __init__(self, base_url, subscription_id, resource_group, workspace_name, cert_file=None):
        self._base_url = base_url
        self._subscription_id = subscription_id
        self._resource_group = resource_group
        self._workspace_name = workspace_name
        self._cert_file = cert_file

    @property
    def workspaces_url(self):
        """
         :return: dict
         """
        return f"{self._base_url}/admin/userinfo/workspaces?subscriptionId={self._subscription_id}" \
            f"&resourceGroupName={self._resource_group}&workspaceName={self._workspace_name}" \
            f"&tenantId={DefaultConstants.TENANT_ID}"

    @property
    def headers(self):
        """
        Get headers of the HTTP request

        :return: dict
        """
        return {
            'Content-Type': 'application/json'
        }

    def _send_request(self, url, data=None, method=None):
        """
        Send HTTP request to admin api

        :param url: str
        :param data:
        :param method: str, HTTP request methods
        :return: str
        """
        if data is not None:
            data = json.dumps(data, cls=EnhancedJsonEncoder).encode()

        if method is None:
            method = 'GET' if data is None else 'POST'

        req = Request(url, data, self.headers, method=method)
        # need client certificate to communicate with server
        context = _init_ssl_context(cert_file=self._cert_file)
        try:
            with urlopen(req, context=context) as res:
                body = res.read().decode()
                return body
        except HTTPError as e:
            raise AlghostRuntimeError(f"Failed while performing {method} {url}: {e.reason}") from e
        except BaseException as e:
            raise AlghostRuntimeError(f"Failed while performing {method} {url}") from e

    def get_workspace_id(self):
        """
        :return: str
        """
        ret = self._send_request(self.workspaces_url)
        return json.loads(ret)

      
def get_module_registry(global_or_workspace, work_space=None):
    """
    Get ModuleRegistry instance in global workspace or a specified workspace

    :param global_or_workspace: str, 'global' or 'workspace'
    :param work_space: Workspace
    :return: ModuleRegistry
    """

    server_conf = ServerConf('int')
    if global_or_workspace == 'global':
        return ModuleRegistry(
            base_url=server_conf.url,
            workspace_id=server_conf.workspace_id,
            cert_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), DefaultConstants.CERT_FILE_NAME)
        )
    elif global_or_workspace == 'workspace':
        workspace_registry = WorkspaceRegistry(
            base_url=server_conf.url,
            subscription_id=work_space.subscription_id,
            resource_group=work_space.resource_group,
            workspace_name=work_space.name,
            cert_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), DefaultConstants.CERT_FILE_NAME)
        )
        return ModuleRegistry(
            base_url=server_conf.url,
            workspace_id=workspace_registry.get_workspace_id(),
            cert_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), DefaultConstants.CERT_FILE_NAME)
        )
    else:
        raise ValueError(f"Argument 'global_or_workspace' must be 'global' or 'workspace', got {global_or_workspace}")


def get_default_workspace():
    return Workspace.get(
        name=DefaultConstants.WORKSPACE_NAME,
        subscription_id=DefaultConstants.SUBSCRIPTION_ID,
        resource_group=DefaultConstants.RESOURCE_GROUP
    )


def list_modules(workspace, show_summary=True):
    """
    List all global modules and custom modules in workspace.

    :param workspace: Workspace
    :return:
    """
    # global_module_registry = get_module_registry('global')
    workspace_module_registry = get_module_registry('workspace', workspace)
    # global_module_specs = global_module_registry.list_modules()
    workspace_module_specs = workspace_module_registry.list_modules()
    module_list = [Module(spec_dct=spec) for spec in workspace_module_specs]
    number_of_global_modules = len([m for m in module_list if m.is_global_module])
    if show_summary:
    	print("Listing modules ...\n")
    	print(f"Found {len(module_list)} modules, among which {number_of_global_modules} global modules, "
         	f"{len(module_list)-number_of_global_modules} workspace modules.")
    return module_list
  
    
def register_module(workspace, git_url, branch_or_tag=None, commit_hash=None, spec_file_name=None, pip_lock=False):
    resource_id = f'/subscriptions/{workspace.subscription_id}/resourceGroups/{workspace.resource_group}' \
                  f'/providers/Microsoft.MachineLearningServices/workspaces/{workspace.name}'

    _register_custom_module(git_url, branch_or_tag=branch_or_tag, commit_hash=commit_hash, spec_file_name=spec_file_name,
                           pip_lock=pip_lock, resource_id=resource_id, cert_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), DefaultConstants.CERT_FILE_NAME),
                           conf='int', clean=True)


class Parameter:
    """
    Parameter of a module

    """

    def __init__(self, name, description, child_params):
        self.name = name
        self.description = description
        self.child_params = child_params
        self.value = None  # to be set by code
        self._assigned = False  # True after this parameter is assigned

    @staticmethod
    def from_dct(dct: dict):
        """
        Initialize a Parameter instance from a dict

        :param dct: dict
        :return: Parameter instance
        """
        # Get child parameter list
        child_params = []
        if isinstance(dct.get('ModeValuesInfo'), list):
            for child_param in dct.get('ModeValuesInfo'):
                if isinstance(child_param, dict):
                    for child_param_dict in child_param.get('Value').get('Parameters'):
                        child_params.append(Parameter.from_dct(child_param_dict))

        return Parameter(
            name=dct.get('FriendlyName'),
            description=dct.get('Description'),
            child_params=child_params
        )

    def assign(self, value):
        """
        Assign value to port

        """
        self.value = value
        self._assigned = True

    @property
    def assigned(self):
        """
        If a parameter has been assigned

        :return: bool
        """
        return self._assigned


class Port:
    """
    Input/output-port of a module

    """

    def __init__(self, name, type_, description):
        self.name = name
        self.type = type_
        self.description = description
        self.value = None  # to be set by code
        self._connected = False  # True after this port is connected

    @staticmethod
    def from_dct(dct: dict):
        """
        Initialize a Port instance from a dict

        :param dct: dict
        :return: Port instance
        """
        return Port(
            name=dct.get('FriendlyName'),
            type_=dct.get('Type'),
            description=dct.get('Description')
        )

    def prepare(self):
        """
        Prepare a Port instance to be connected by assigning a PipelineData instance to its value

        """
        def _regular_name(port_name):
            # AML Service does not allow name with spaces. Replace them with underscore.
            return '_'.join(port_name.split())

        if not self.prepared:
            conn = PipelineData(_regular_name(self.name))
            self.value = conn

    @property
    def prepared(self):
        """
        If a port is prepared for connection

        :return: bool
        """
        return self.value is not None

    @property
    def connected(self):
        """
        If a parameter has been connected

        :return: bool
        """
        return self._connected

    def connect(self, another_port):
        """
        Connect a port to another

        :param another_port: Port
        """
        if not another_port:
            raise ValueError(f"Cannot connect to a empty port")
        if not another_port.prepared:
            raise ValueError(f"Port({another_port.name}) is not ready yet")

        self.value = another_port.value
        self._connected = True


class Module:
    """
    Module class.

    Module class represents the interface, including input/output ports, parameters of a module
    as well as its conda dependencies and command line interface
    """

    @classmethod
    def create(cls, work_space, module_name):
        """
        Create a Module instance of a module, which is registered under work_space

        :param work_space: Workspace
        :param module_name: str
        :return: Module
        """

        for m in list_modules(work_space, show_summary=False):
            if m.name == module_name:
            	print(f"Creating {m}")
            	return m

    def __init__(self, spec_dct):
        """
        Initialize a Module instance

        :param spec_dct: dict
        """
        self._dct = spec_dct
        self._input_ports = [Port.from_dct(p) for p in self._get_value('ModuleInterface/InputPorts')]
        self._output_ports = [Port.from_dct(p) for p in self._get_value('ModuleInterface/OutputPorts')]
        self._params = [Parameter.from_dct(p) for p in self._get_value('ModuleInterface/Parameters')]

        for p in self._output_ports:
            p.prepare()
    
    def __str__(self):
        return f"Module {self.name} (version: {self.batch_version}, owner: {self.owner}, " \
            f"created date: {self.created_date})"

    def _get_value(self, key_path):
        if not key_path:
            raise ValueError("key_path must not be empty")
        if not self._dct:
            raise ValueError("dct is empty")

        segments = key_path.split('/')

        walked = []

        cur_obj = self._dct
        for seg in segments:
            if cur_obj is None:
                raise ValueError(f"Missing {'/'.join(walked)} block in dict")
            if not isinstance(cur_obj, dict):
                raise ValueError(f"Block {'/'.join(walked)} cannot contain a child")

            cur_obj = cur_obj.get(seg)
            walked.append(seg)

        if cur_obj is None:
            raise ValueError(f"Missing {'/'.join(walked)} block in dict")
        return cur_obj

    def assign_parameters(self, name_value_dict):
        """
        Assign parameters from a dict

        :param name_value_dict: dict
        """
        for name, value in name_value_dict.items():
            self.params[name].assign(value)
            # Append child parameters to the parameter list
            for child in self.params[name].child_params:
                self._params.append(child)

    @property
    def name(self):
        """
        Get module name

        :return: str
        """
        return self._get_value('Name')
    
    @property
    def batch_version(self):
        """
        Get module version

        :return: str
        """
        return self._get_value('Batch')

    @property
    def created_date(self):
        """
        Get module created date

        :return: str
        """
        dt = self._get_value('CreatedDate')  # dt has the format of "/Date(1562059179924)"
        # Get the str between parenthesis
        dt_str = dt[dt.find("(")+1:dt.find(")")]
        # Get the 10-digit int number
        dt_int = int(int(dt_str)/1000)
        return datetime.utcfromtimestamp(dt_int)

    @property
    def owner(self):
        """
        Get module owner

        :return: str
        """
        return self._get_value('Owner')
   
    @property
    def is_global_module(self):
        """
        If global module
        
        :return bool
        """
        return self._get_value('Id').startswith(DefaultConstants.GLOBAL_MODULE_ID)

    @property
    def description(self):
        """
        Get module description

        :return: str
        """
        return self._get_value('Description')

    @property
    def input_ports(self):
        """
        Get module input ports

        :return: dict, eg: {port name: Port instance}
        """
        return {p.name: p for p in self._input_ports}

    @property
    def input_refs(self):
        """
        Get the list of connected input ports

        :return: list
        """
        return [p.value for p in self._input_ports if p.connected]

    @property
    def params(self):
        """
        Get module parameters

        :return: dict, eg: {parameter name: Parameter instance}
        """
        return {p.name: p for p in self._params}

    @property
    def output_ports(self):
        """
        Get module output ports

        :return: dict, eg: {port name: Port instance}
        """
        return {p.name: p for p in self._output_ports}

    @property
    def output_refs(self):
        """
        Get the list of output ports

        :return: list
        """
        return [p.value for p in self._output_ports]

    @property
    def conda_dependencies(self):
        """
        Get module conda dependencies

        :return: CondaDependencies instance
        """
        cd = CondaDependencies()
        for c in self._get_value('CondaDependencies/CondaChannels'):
            cd.add_channel(c)
        for c in self._get_value('CondaDependencies/CondaPackages'):
            cd.add_conda_package(c)
        for p in self._get_value('CondaDependencies/PipPackages'):
            cd.add_pip_package(p)
        for p in self._get_value('CondaDependencies/PipOptions'):
            cd.set_pip_option(p)
        return cd

    @property
    def command(self):
        """
        Get first part of command line

        :return: list, eg: ['python', '-m', 'script.train']
        """
        return self._get_value('CommandLineEntry/Command')

    @property
    def args(self):
        """
        Get command line arguments

        :return: list
        """
        def name_to_value(name):
            for port_name, port in self.input_ports.items():
                if port.connected and port_name == name:
                    return port.value

            for param_name, param in self.params.items():
                if param.assigned and param_name == name:
                    return param.value

            for port_name, port in self.output_ports.items():
                if port_name == name:
                    return port.value
            return None

        raw_args = self._get_value('CommandLineEntry/OptionNames')
        args = []
        for raw_arg in raw_args:
            arg = name_to_value(raw_arg.get('Key'))
            if arg:
                args.append(raw_arg.get('Value'))
                args.append(arg)
        return args

    @property
    def command_and_args(self):
        """
        Get full command line arguments

        :return: list
        """
        return self.command + self.args


class ModuleStep(PythonScriptStep):
    SCRIPT_FILE_NAME = 'invoker.py'

    def __init__(self, module, global_run_config=None, allow_reuse=False):
        """
        Initialize a ModuleStep instance

        :param module: Module
        :param allow_reuse: bool
        :param run_config: RunConfiguration
        """
        self._comp = module

        run_config = self.get_run_config(global_run_config)

        self.source_directory = os.path.dirname(os.path.abspath(__file__))

        print(f"== Creating ModuleStep: name={self._comp.name}\n"
              f"   arguments={self._comp.command_and_args}\n"
              f"   inputs={self._comp.input_refs}\n"
              f"   outputs={self._comp.output_refs}\n"
              )

        super().__init__(
            name=self._comp.name,
            source_directory=self.source_directory,
            script_name=self.SCRIPT_FILE_NAME,
            arguments=self._comp.command_and_args,
            inputs=self._comp.input_refs,
            outputs=self._comp.output_refs,
            compute_target=run_config.target,
            allow_reuse=allow_reuse,
            runconfig=run_config
        )

    def get_run_config(self, global_run_config):
        """
        Get RunConfiguration instance with its base_image to be DEFAULT_GPU_IMAGE

        :return: RunConfiguration
        """
        run_config = RunConfiguration(conda_dependencies=self._comp.conda_dependencies)
        run_config.target = global_run_config.target
        run_config.environment.docker.enabled = global_run_config.environment.docker.enabled
        run_config.environment.docker.base_image = global_run_config.environment.docker.base_image
        run_config.environment.docker.gpu_support = global_run_config.environment.docker.gpu_support

        return run_config


def run_pipeline(steps, experiment_name, workspace=None):
    """
    Run pipeline on AzureML service

    :param steps: ModuleStep
    :param experiment_name: str
    :param workspace: Workspace
    """
    if not workspace:
        workspace = get_default_workspace()

    exp = Experiment(workspace=workspace, name=experiment_name)
    pipeline = Pipeline(workspace=workspace, steps=steps)
    pipeline.validate()

    run = exp.submit(pipeline)
    run.wait_for_completion(show_output=True)
    run.get_metrics()
