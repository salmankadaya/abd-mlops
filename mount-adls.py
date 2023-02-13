# Databricks notebook source
# We set the values of our databricks scope name and azure storage account.
databricks_scope_name = "databricks-kv"
storage_account_name = "rgdevsa"

# COMMAND ----------

# We use dbutils to list our secrets name because we need to send them as parameter to the 
# dbutils.secrets.get()
dbutils.secrets.list(databricks_scope_name)

# COMMAND ----------

# We assign the values of our secrets to variables.
client_id     = dbutils.secrets.get(databricks_scope_name, "adb-client-id")
tenant_id     = dbutils.secrets.get(databricks_scope_name, "adb-ten-id")
client_secret = dbutils.secrets.get(databricks_scope_name, "adb-sp-id")

# COMMAND ----------

# To access to our data in Azure Storage securely we are going to use OAuth 2.0 with Azure Active Directory
# We need to set the following configuration
configs = {
    "fs.azure.account.auth.type": "OAuth",
    "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
    "fs.azure.account.oauth2.client.id": f"{client_id}",
    "fs.azure.account.oauth2.client.secret": f"{client_secret}",
    "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"
}

# COMMAND ----------

# dbutisl.fs.mount: Mounts the specified source directory into DBFS at the specified mount point.
def mount_adls(container_name):
    dbutils.fs.mount(
        source=f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/",
        mount_point=f"/mnt/{storage_account_name}/{container_name}",
        extra_configs = configs
    )

# COMMAND ----------

# Mount Bronze Container
mount_adls("bronze")

# Mount Silver Container
mount_adls("silver")

# Mount Gold Container
mount_adls("gold")


# COMMAND ----------

# List all mounts on databricks
# You must see your bronze, silver and gold contianers and other ones that are by default from databricks.
dbutils.fs.mounts()
