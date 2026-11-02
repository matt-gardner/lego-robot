# AWS Spot GPUs: checklist

Explanations and costs: [detailed guide](AWS_SPOT_OPERATIONS.md).
Never paste passwords, keys, or MFA codes into chat. Do not launch a VM yet.

## 1. You: find or create an account

Try [AWS sign-in](https://console.aws.amazon.com/); for work funding, ask your administrator first.
If needed, [create an account](https://signin.aws.amazon.com/signup?request_type=register): choose **advanced signup**, **Paid plan**, and **Basic support**. Complete verification and wait for activation.

## 2. You: secure it and set an alert

Account menu -> **Security credentials** -> enable **MFA**.
**Billing -> Budgets** -> create a monthly email alert, e.g. $50. This is not a spending cap.

## 3. Paste: set up local login

```text
Read docs/AWS_SPOT_OPERATIONS.md. I can sign into the AWS console.
Help me set up a non-root, short-lived CLI login; do not assume SSO or an existing profile. No paid resources, uploads, or IAM changes without approval.
```

## 4. You: request GPU quota

Select **Oregon** -> **Service Quotas -> Amazon EC2**.
Request **8 vCPUs** for **All G and VT Spot Instance Requests**. Wait for approval.

## 5. Paste: prepare a plan

```text
Prepare AWS Spot support following docs/AWS_SPOT_OPERATIONS.md: Oregon, one g6.2xlarge or g5.2xlarge. Check live prices/quota; implement and locally test missing AWS recovery and cost controls. Propose price, storage, session, and total-hour limits with an itemized bill. No paid actions, uploads, or changes to Mac/Azure training.
```

## 6. Review the plan, then paste: qualify

```text
I approve the reviewed Oregon plan and required project-data uploads within its stated price, storage, session, and total-hour limits. Deploy and qualify CUDA/EGL, batch128/microbatch16, checkpoint restore, interruption recovery, and shutdown. No full training yet, On-Demand fallback, or changes to Mac/Azure. Stop compute afterward.
```

## 7. After qualification, paste: train

```text
Start the reviewed new CUDA SFT run within the remaining approved limits. Preserve the recipe and frozen40 evaluation; use a distinct output. Verify a finite update, uploaded recovery checkpoint, and private dashboard. Keep backups, recovery, shutdown limits, and one-worker enforcement active.
```

## 8. When finished, paste: stop

```text
Save and verify recovery artifacts; cancel replacement requests and stop AWS compute. Report retained storage/IP costs. Do not delete recovery assets without approval.
```

## 9. Optional later: eastern fallback

Repeat step 4 in **Ohio**, then paste:

```text
Plan an Ohio fallback with one active trainer across both regions. Itemize additional costs; no deployment, replication, or increased limits before approval.
```
