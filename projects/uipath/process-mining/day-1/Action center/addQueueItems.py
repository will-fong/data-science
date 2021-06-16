import requests
import json
import os
import sys
from dotenv import load_dotenv


def getAuthKey(tenant):
    instance_type = os.getenv('ORCH_INSTANCE_TYPE')

    # Check whether the instance type is filled correctly in the configuration file.
    if instance_type != '1' and instance_type != '2':
        sys.exit("ORCH_INSTANCE_TYPE is not filled correctly in the configuration file.")

    if instance_type == "1":
        return getOnPremAuthKey(tenant)
    elif instance_type == "2":
        return getCloudAuthKey()


def getOnPremAuthKey(tenant):
    url = os.getenv('ORCH_URL')
    auth_endpoint_prem = '/api/Account/Authenticate'

    username = os.getenv('ORCH_USERNAME')
    password = os.getenv('ORCH_PASSWORD')

    # Check whether the URL is filled in the configuration file.
    if url == '':
        sys.exit("ORCH_URL is not filled in the configuration file.")

    # Check whether the username and password are filled in the configuration file.
    if username == '' or password == '':
        sys.exit("ORCH_USERNAME and/or ORCH_PASSWORD are not filled in the configuration file.")

    body = {
        "tenancyName": tenant,
        "usernameOrEmailAddress": username,
        "password": password
    }

    # Get the authentication key.
    try:
        req = requests.post(url + auth_endpoint_prem, body)
    except requests.ConnectionError:
        sys.exit("Request for the authentication key results in a connection error for url: " + url + ".")

    resJson = req.json()

    if req.status_code == 200:
        return resJson["result"]
    else:
        sys.exit("Request for the authentication key is not successful with status code: " + str(req.status_code) + ".")


def getCloudAuthKey():
    auth_endpoint_cloud = 'https://account.uipath.com/oauth/token'

    client_id = os.getenv('ORCH_CLIENT_ID')
    user_key = os.getenv('ORCH_USER_KEY')

    # Check whether the client ID and user key are filled in the configuration file.
    if client_id == '' or user_key == '':
        sys.exit("ORCH_CLIENT_ID and/or ORCH_USER_KEY are not filled in the configuration file.")

    body = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": user_key
    }

    # Get the authentication key.
    req = requests.post(auth_endpoint_cloud, body)
    resJson = req.json()

    if req.status_code == 200:
        return resJson["access_token"]
    else:
        sys.exit("Request for the authentication key is not successful with status code: " + str(req.status_code) + ".")


def checkItemInput(item_input):
    # First check whether name is present to use it in the other error messages.
    if "name" not in item_input:
        sys.exit("Mandatory field 'name' is not defined for a queue item.")

    if "folderID" not in item_input:
        sys.exit("Mandatory field 'folderID' is not defined for a queue item in queue: " + item_input["name"] + ".")

    if "reference" not in item_input:
        sys.exit("Mandatory field 'reference' is not defined for a queue item in queue: " + item_input["name"] + ".")


def getQueues(item_input, token, tenant):
    url = os.getenv('ORCH_URL')
    queue_definitions_endpoint = '/odata/QueueDefinitions'

    headers = {
        "Content-Type": "application/json",
        "X-UIPATH-TenantName": tenant,
        "Authorization": "Bearer " + token,
        "X-UIPATH-OrganizationUnitId": item_input["folderID"]
    }

    # Get the queue names of the available queues in the folder.
    req = requests.get(url + queue_definitions_endpoint + '?$select=Name', headers=headers)
    resJson = req.json()

    if req.status_code != 200:
        sys.exit("Request to create queue item is not successful with status code: " + str(req.status_code)
                 + ".\n" + resJson["message"]
                 )

    # Create a list of all queue names that are available in the folder.
    queue_names = []
    for queue in resJson["value"]:
        queue_names.append(queue["Name"])

    return queue_names


def generateQueueItemData(item_input):
    queue_item_data = {"itemData": {}}

    # Add mandatory fields
    queue_item_data["itemData"]["Name"] = item_input["name"]
    queue_item_data["itemData"]["Reference"] = item_input["reference"]

    # Add optional fields, when available.
    if "priority" in item_input:
        queue_item_data["itemData"]["Priority"] = item_input["priority"]

    if "deadline" in item_input:
        queue_item_data["itemData"]["DueDate"] = item_input["deadline"]

    if "postpone" in item_input:
        queue_item_data["itemData"]["DeferDate"] = item_input["postpone"]

    # Add specific data.
    queue_item_data["itemData"]["SpecificContent"] = {}

    for key in item_input.keys():
        if key not in ["folderID", "name", "reference", "priority", "deadline", "postpone"]:
            queue_item_data["itemData"]["SpecificContent"][key] = item_input[key]

    return queue_item_data


def createQueueItem(item, queues_in_folder, token, tenant):
    url = os.getenv('ORCH_URL')
    add_queue_item_endpoint = '/odata/Queues/UiPathODataSvc.AddQueueItem'

    # Check whether the URL is filled in the configuration file.
    if url == '':
        sys.exit("ORCH_URL is not filled in the configuration file.")

    # Check whether all mandatory fields to define a queue item are present.
    item_input = json.loads(item)
    checkItemInput(item_input)

    # Get the available queues in the folder in which the item should be added.
    # Only needed when the folder is not encountered before.
    if item_input["folderID"] not in queues_in_folder:
        queues_in_folder[item_input["folderID"]] = getQueues(item_input, token, tenant)

    # Check whether the queue in which the item should be added is available.
    if item_input["name"] not in queues_in_folder[item_input["folderID"]]:
        sys.exit("No queue available in folder " + item_input["folderID"] + " with name: " + item_input["name"] + ".")

    # Generate the item data.
    item_data = generateQueueItemData(item_input)

    headers = {
        "Content-Type": "application/json",
        "X-UIPATH-TenantName": tenant,
        "Authorization": "Bearer " + token,
        "X-UIPATH-OrganizationUnitId": item_input["folderID"]
    }

    # Create the queue item in the orchestrator.
    req = requests.post(url + add_queue_item_endpoint, json.dumps(item_data), headers=headers)
    resJson = req.json()

    # Status code 201 indicates a successful creation of a queue item.
    # Status code 409, but more specifically with the message 'Error creating Transaction. Duplicate Reference.',
    # indicates that the queue item is not created for a queue with unique references.
    # It is intended behavior to not create this queue item and continue with the execution of the script.
    if req.status_code != 201 and resJson["message"] != 'Error creating Transaction. Duplicate Reference.':
        sys.exit("Request to create queue item is not successful with status code: " + str(req.status_code)
                 + ".\n" + resJson["message"]
                 )


if __name__ == '__main__':
    # Get location of the directory in which the 'actioncenter' folder is placed.
    directory = os.path.abspath('.')
    path = "/".join(directory.split('\\')[:-4])

    # Setting in the connector to specify for which tenant the actions are created.
    tenant = sys.argv[2]

    # Check whether tenant is set in the connector.
    if tenant == '':
        sys.exit("Tenant is not set in the connector.")

    # Check whether the configuration file for the tenant can be found.
    if not os.path.exists(path + "/actioncenter/" + tenant + ".env"):
        sys.exit("Could not find the configuration file '<PLATFORMDIR>/actioncenter/" + tenant + ".env'.")

    # Load the orchestrator configuration file for the tenant.
    load_dotenv(path + "/actioncenter/" + tenant + ".env")

    # Get the authentication key for the tenant.
    token = getAuthKey(tenant)

    # Available queues in the orchestrator per folder.
    # Each queue item contains the information in which folder it should be created.
    # Check the available queues in a folder once for each newly encountered folder.
    queues_in_folder = {}

    # Read the input from the connector.
    queue_items = sys.argv[1]

    # Create the queue items.
    with open(queue_items) as f:
        for item in f:
            createQueueItem(item, queues_in_folder, token, tenant)
