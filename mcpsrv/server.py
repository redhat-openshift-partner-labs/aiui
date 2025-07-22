from toolbox_core import ToolboxSyncClient

with ToolboxSyncClient("http://127.0.0.1:5000") as tbsclient:
    tools = tbsclient.load_toolset("partner_labs")