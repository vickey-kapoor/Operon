<#
.SYNOPSIS
Quiets Microsoft Edge for the Hybrid benchmark (KAN-73): no first-run pages, pop-ups
or helpers over the pages the agent works on.

.DESCRIPTION
Hybrid tasks have the agent open Edge on the Windows desktop and use the benchmark's
local site. Edge's first-run wizard, "restore pages" prompt, password and address
autofill, shopping assistant and sidebar would sit over those pages or fill forms for
the agent, so these machine-wide policies turn them off. They apply to every account
on the machine, the runner's included.

Run it elevated. The "Live desktop benchmark" workflow runs it on GitHub's hosted
runner before each Hybrid run. On the Azure VM the runner account is not an
administrator, so run it once yourself, over RDP as the VM's admin account
(docs/azure-desktop-runner.md). Running it again changes nothing.
#>
$ErrorActionPreference = "Stop"

$key = "HKLM:\SOFTWARE\Policies\Microsoft\Edge"
New-Item -Path $key -Force | Out-Null
$policies = [ordered]@{
    HideFirstRunExperience        = 1  # no welcome wizard on first start
    AutoImportAtFirstRun          = 4  # import nothing from other browsers
    HideRestoreDialogEnabled      = 1  # no "restore pages" bubble after the benchmark stops Edge
    RestoreOnStartup              = 5  # start on the new tab page
    DefaultBrowserSettingEnabled  = 0  # don't ask to be the default browser
    BrowserSignin                 = 0  # no sign-in prompts
    SyncDisabled                  = 1
    PromotionalTabsEnabled        = 0
    ShowRecommendationsEnabled    = 0
    SpotlightExperiencesAndRecommendationsEnabled = 0
    PasswordManagerEnabled        = 0  # no "save password?" over the account pages
    AutofillAddressEnabled        = 0  # the agent fills checkout and support forms itself
    AutofillCreditCardEnabled     = 0
    EdgeShoppingAssistantEnabled  = 0  # no coupon and price pop-ups on the shop
    HubsSidebarEnabled            = 0
    TranslateEnabled              = 0
    StartupBoostEnabled           = 0  # Edge exits when its windows close
    BackgroundModeEnabled         = 0
}
foreach ($name in $policies.Keys) {
    New-ItemProperty -Path $key -Name $name -Value $policies[$name] -PropertyType DWord -Force | Out-Null
}
Write-Output "Edge policies set under $key ($($policies.Count) values)."
