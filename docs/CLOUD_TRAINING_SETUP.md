# Azure Spot training: your setup checklist

## 1. Create an account and choose the subscription

Go to [Azure pay-as-you-go signup](https://azure.microsoft.com/pricing/purchase-options/pay-as-you-go/).
Sign in with a Microsoft account (or create one), then complete identity and
payment verification.

Choose **Pay-As-You-Go**, offer **003P**. This is explicitly supported for
[Azure Spot VMs](https://learn.microsoft.com/azure/virtual-machines/spot-vms#limitations).
Do not rely on a free trial or Azure for Students subscription for this setup;
upgrade to Pay-As-You-Go first. For a Free Trial, open **Subscriptions** -> your
subscription -> **Overview** and select **Upgrade subscription** in the top
command bar. If that button is missing, use the upgrade banner at the top of the
page. Follow the payment/verification prompts; the **Basic** support plan is
enough to start. See [Microsoft's upgrade instructions](https://learn.microsoft.com/azure/cost-management-billing/manage/upgrade-azure-subscription).
An existing Enterprise Agreement subscription
also supports Spot, subject to your organization's permissions.

If quota requests still say **"Free trials are not eligible"** after upgrading,
confirm the upgrade completed for the same subscription selected on the quota
page. Reopen the portal in a fresh session. If that subscription's offer still
shows Free Trial despite a confirmed upgrade, contact **Help + support** for a
subscription/billing issue; this is an eligibility problem, not a quota denial.

Open [portal.azure.com](https://portal.azure.com), search **Subscriptions**, select
that subscription, and copy its **Subscription ID**.

## 2. Sign in on this Mac

```sh
brew install azure-cli   # only if az is not installed
az login
az account set --subscription YOUR-SUBSCRIPTION-ID
```

## 3. Request Spot quota

If the quota page says **"The selected provider is not registered"**, first open
**Subscriptions** -> your subscription -> **Resource providers**. Search for
**Microsoft.Compute**, select it, and click **Register**. Wait until its status
is **Registered**, then return to Quotas and click **Refresh**.

In the portal, search **Quotas** -> **Compute**, choose your subscription and
region, and request **24 Total Regional Spot vCPUs**.
Our initial GPU is `Standard_NC24ads_A100_v4` (one 80 GB A100).
Start with **West US 3 (`westus3`)**. Read-only SKU checks on October 31, 2026
showed no subscription restrictions for this A100 size there, with zones **1**
and **3** listed. West US 2 and East US 2 were restricted for the CLI-selected
subscription; East US listed zone 3 but restricted zonal use. These restrictions
can change and do not measure free GPU capacity. West US 3's Spot quota was
only **3 vCPUs**, so request **24 in West US 3**, not just in another region.
The [Azure Retail Prices API](https://prices.azure.com/api/retail/prices)
checked on October 31, 2026 lists Linux Spot `Standard_NC24ads_A100_v4` at
**$0.67877/hour** in West US 2, West US 3, East US, and East US 2.
That is about **$13.58 for 20 GPU-hours**, excluding disks, backups, and networking.
West US 3 has the same lowest published price; there is no price advantage to
starting in the eastern regions.
Published pricing does not establish available Spot capacity or eviction risk.
An authenticated Resource Graph query returned no A100 eviction-rate rows,
including with a relaxed SKU-name filter; no eviction-rate estimate was verified.
Check the portal's Spot pricing/eviction comparison before deploying. Successful
allocation is the final capacity check and creates billable resources; do not
attempt it without approval.
Training runs entirely on the VM, so capacity and price matter more than
distance from the laptop. Keep the VM, retained disk, and quota region aligned.
If the quota cannot be increased through the portal, open an Azure support request.
Approved quota does not guarantee that this GPU has Spot capacity in that region.

## 4. Confirm access

You must be able to create resources, assign roles, and create deletion locks in a
dedicated resource group. If this is an organization-managed subscription, ask
your administrator for **Contributor + User Access Administrator**, or have an
Owner perform initial deployment. The local upload identity will also need
**Storage Blob Data Contributor** on the backup storage account after creation.

## 5. Set a budget alert and choose limits

Portal -> **Cost Management** -> **Budgets**: create a monthly budget and email
alerts at amounts you are comfortable with.

**Budget alerts are not a hard spending cap.** Choose a maximum GPU hourly price,
session duration, and total GPU hours before deployment. Disks and backup storage
continue billing when the GPU is stopped.

That is the manual prerequisite checklist. Config, SSH keys, deployment, GPU
setup, and checkpoint migration can be handled from this project afterward,
with explicit approval before creating paid resources. No VM is created by
following this checklist; do not create one manually.

The [operations reference](CLOUD_TRAINING_OPERATIONS.md) contains deployment,
qualification, pause/resume, recovery, and deletion details; it is not required
reading to complete these prerequisites.
