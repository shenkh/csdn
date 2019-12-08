 # [Custom install or change Microsoft Office with Office Deployment Tool](<https://www.tenforums.com/tutorials/123233-custom-install-change-microsoft-office-office-deployment-tool.html>)

![information](https://www.tenforums.com/images/infosmall10.png)   **Information**

In earlier versions of **Microsoft Office**, users could select Install options and customize the installation by installing only selected Office applications, or select an application to be installed on first use, or completely disable an application. Later on, this could be changed, other Office applications installed or disabled. 

In **Office 2016** and later, including **Office 365**, this is no longer possible. Full Office suite of applications will be installed, be it an MSI installation from ISO or Click-to-Run installation, and it is no longer possible to remove or add individual Office applications.

The **Office Deployment Tool** (ODT) is a command-line tool that you can use to deploy (install) Office to your computers. Using ODT, you can select which Office suite applications will be installed or removed. You can also add or remove additional languages and edit various options.

ODT needs a configuration script (XML file), which can be created with **Office Customization Tool** or manually.

This tutorial will show how to create or modify a configuration script and use it with ODT to custom install Office or modify an existing installation.

![Custom install or change Microsoft Office with Office Deployment Tool](https://www.tenforums.com/IMAGES/OPTl.PNG) Contents ![Custom install or change Microsoft Office with Office Deployment Tool](https://www.tenforums.com/IMAGES/OPTR.PNG)

 Click links to jump to any part

| [**Part One:**](https://www.tenforums.com/tutorials/123233-custom-install-change-microsoft-office-office-deployment-tool.html#Part1) | **Microsoft Office Editions**                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**Part Two:**](https://www.tenforums.com/tutorials/123233-custom-install-change-microsoft-office-office-deployment-tool.html#Part2) | **Configure install options with Office Customization Tool** |
| [**Part Three:**](https://www.tenforums.com/tutorials/123233-custom-install-change-microsoft-office-office-deployment-tool.html#Part3) | **Configure and edit install options manually**              |
| [**Part Four:**](https://www.tenforums.com/tutorials/123233-custom-install-change-microsoft-office-office-deployment-tool.html#Part4) | **Run Setup to install Office**                              |
| [**Part Five:**](https://www.tenforums.com/tutorials/123233-custom-install-change-microsoft-office-office-deployment-tool.html#Part5) | **Change existing Office installation**                      |

Please notice: screenshots can be expanded by clicking them.



![Custom install or change Microsoft Office with Office Deployment Tool](https://www.tenforums.com/IMAGES/OPTl.PNG) Part One ![Custom install or change Microsoft Office with Office Deployment Tool](https://www.tenforums.com/IMAGES/OPTR.PNG)

##  Microsoft Office Editions

1.1) To use ODT, configuration script requires information about which edition (Product ID) to install. Installing wrong edition might make activating Office impossible.

Always use the correct Product ID corresponding with your O365 subscription plan, product key or volume license.

1.2)  Table of editions. In **Activation** column available activation methods for each edition:

- **P** = Retail product key
- **S** = O365 Personal, Home, Business or Enterprise subscription / license
- **V** = Volume licensing (KMS or MAK)

| **Edition**                 | **Product ID**          | **Activation** |
| --------------------------- | ----------------------- | -------------- |
| O365 Home and Student       | HomeStudentRetail       | P              |
| O365 Home and Business      | HomeBusinessRetail      | P              |
| Office 2016 Personal        | PersonalRetail          | P              |
| O365 Personal               | O365PersonalRetail      | S              |
| O365 Home                   | O365HomePremRetail      | S              |
| O365 ProPlus                | O365ProPlusRetail       | V              |
| O365 Enterprise E3, E4, E5  | O365ProPlusRetail       | S V            |
| O365 Business               | O365BusinessRetail      | S              |
| O365 Business Premium       | O365BusinessRetail      | S              |
| O365 Small Business Premium | O365SmallBusinessRetail | S              |
| O365 ProPlus 2019 VL        | ProPlus2019Volume       | V              |
| O365 Standard 2019 VL       | Standard2019Volume      | V              |
| Office 2019 Home & Student  | HomeStudent2019Retail   | P              |
| Office 2019 Home & Business | HomeBusiness2019Retail  | P              |
| Office 2019 Personal        | Personal2019Retail      | P              |
| Office 2019 Professional    | Professional2019Retail  | P              |
| Office 2019 ProPlus         | ProPlus2019Retail       | P              |



![Custom install or change Microsoft Office with Office Deployment Tool](https://www.tenforums.com/IMAGES/OPTl.PNG) Part Two ![Custom install or change Microsoft Office with Office Deployment Tool](https://www.tenforums.com/IMAGES/OPTR.PNG)

## Configure install options with Office Customization Tool

**2.1)** Using ODT requires a configuration file in XML format. When setup is run, it reads installation options from this file. **Office Customization Tool** is a web based application to create ODT configuration files with a few mouse clicks.

**2.2)** Go to [Office Configuration Tool](https://config.office.com/)

**2.3)** Expand **Products and releases**:

![Click image for larger version.   Name:	image.png  Views:	97  Size:	132.0 KB  ID:	216889](https://www.tenforums.com/attachments/tutorials/216889d1544478154t-custom-install-change-microsoft-office-office-deployment-tool-image.png)

**2.4)** Select bit architecture (#1), Office edition from **Office Suites** (#2), which Office applications you want to install and which to exclude (#3), your preferred update channel (#4), and which version you want to install (#5):

![Click image for larger version.   Name:	image.png  Views:	140  Size:	441.8 KB  ID:	216904](https://www.tenforums.com/attachments/tutorials/216904d1544485539t-custom-install-change-microsoft-office-office-deployment-tool-image.png)

![Note](https://www.tenforums.com/images/notesmall10.png)   Note

**Edition selection** (#2 in screenshot):

- Office Customization tool lets you choose **O365 ProPlus**, **O365 Business**, **Office 2019 Standard VL** or **Office 2019 ProPlus VL** edition.
- If you are installing **any other edition**, select **O365 ProPlus** and edit configuration script in **Notepad** when configuration file is ready and downloaded to change edition product ID to correct one. See **step 3.3** about how to edit configuration script.

**Excluded applications** (#3 in screenshot):

- Business and ProPlus editions, exclude **OneDrive Desktop**. OneDrive is built-in and preinstalled in Windows 10 and does not need to be installed with Office.
- **OneDrive (Groove)** is SharePoint. Exclude if not required.
- If your organization has already switched to **Microsoft Teams**, exclude **Skype for Busines**s.

**Channel selection** (#4 in screenshot):

- **Monthly Targeted** = Office Insider, weekly updates, shown as **Channel="Insiders"** in script
- **Monthly** = Monthly updates, shown as **Channel="Monthly"** in script
- **Semi-Annual** = Updated twice a year, in January and July, shown as **Channel="Broad"** in script
- **Semi-Annual Targeted** = Preview of Targeted, twice a year March and September, shown as **Channel="Targeted"** in script

  **2.5)** Click **Next** or expand **Language** . Select the primary installation language, and additional languages you want to install. In additional languages you can select **Full** to install a complete language pack. If a full language pack is not available in your language, select **Partial** to see list of partially translated languages. **Proofing** shows you languages which do not have a full or partial language pack but with Office proofing tools available.

In this example, I selected English as primary language, and Finnish, Swedish and German language packs.  

![Click image for larger version.   Name:	image.png  Views:	41  Size:	164.5 KB  ID:	216921](https://www.tenforums.com/attachments/tutorials/216921d1544488935t-custom-install-change-microsoft-office-office-deployment-tool-image.png)

  **2.6)** Click **Next** or expand **Installation**. Select from where Office will be installed. **Office Content Delivery Network** (CDN) always downloads Office installation files from cloud (recommended, no source path required), **Local source** downloads Office setup files to a network share or local folder, and installation is run from this folder. The share or folder must exist when setup is run.

In case you select **Local source**, setup checks what content is already available on given share or folder, and only downloads what is missing. When setup is run first time, content not excluded in configuration script will be downloaded and saved on given **Source path**. This saves time and bandwidth in case you have multiple computers; use a network share as **Source path**, download is only done once and Office can be deployed from that share to all computers.

The same if you want to modify existing Office installation. Let's say I originally installed Finnish Office, excluding Access and Publisher. Now I want to add Swedish language pack, Access and Publisher to my existing installation. As setup already finds downloaded Office core and most of the applications from my source, a network share, it only needs to download those components I added.
  

![Click image for larger version.   Name:	image.png  Views:	57  Size:	137.0 KB  ID:	216925](https://www.tenforums.com/attachments/tutorials/216925d1544489861t-custom-install-change-microsoft-office-office-deployment-tool-image.png)

  Turn **Show installation to user** off if you want no information or progress bars about running installation shown to user. I recommend always turning it **ON**.

**2.7)** Click **Next** or expand **Update and upgrade**. Select if you want future updates and upgrades being installed from CDN or local source (see 2.6 for difference). Turn automatic updates on or off, select if you want any possible remnants from earlier MSI installation removed before installing Click-to-Run version of Office:  

![Click image for larger version.   Name:	image.png  Views:	136  Size:	199.6 KB  ID:	216929](https://www.tenforums.com/attachments/tutorials/216929d1544490941t-custom-install-change-microsoft-office-office-deployment-tool-image.png)

**2.8)** Click **Next** or expand **Licensing and activation**. In case you selected a volume licensed edition of Office in **step 2.4**, select your volume licencing method. For all editions, select **Automatically accept the EULA**. Do not turn **Autoactivate** on:

![Click image for larger version.   Name:	image.png  Views:	56  Size:	219.1 KB  ID:	216932](https://www.tenforums.com/attachments/tutorials/216932d1544491673t-custom-install-change-microsoft-office-office-deployment-tool-image.png)

**2.9)** Click **Next** or expand **General**. This step is optional, but if you want to you can add your name or name of an organization and remarks to be saved in configuration file:

![Click image for larger version.   Name:	image.png  Views:	57  Size:	91.7 KB  ID:	216933](https://www.tenforums.com/attachments/tutorials/216933d1544492299t-custom-install-change-microsoft-office-office-deployment-tool-image.png)

  **2.10** Click **Next** or expand **Application preferences**. This step is optional, you can pre-set a lot of general Office preferences here if you so prefer.

Click a preference to change it, and set the value as you want to:  

![Click image for larger version.   Name:	image.png  Views:	41  Size:	124.3 KB  ID:	216934](https://www.tenforums.com/attachments/tutorials/216934d1544492715t-custom-install-change-microsoft-office-office-deployment-tool-image.png)

A pop-up opens to let you change preferences:
![Name:  image.png Views: 5758 Size:  124.8 KB](https://www.tenforums.com/attachments/tutorials/216935d1544492872-custom-install-change-microsoft-office-office-deployment-tool-image.png)

**2.11)** All done. You can now export (download) the configuration file:

![Click image for larger version.   Name:	image.png  Views:	64  Size:	503.9 KB  ID:	216936](https://www.tenforums.com/attachments/tutorials/216936d1544493052t-custom-install-change-microsoft-office-office-deployment-tool-image.png)

**2.12)** In this example, I created a configuration script to install O365 ProPlus, excluding OneDrive, Publisher and Skype for Business (shown as *"Lync"* in script). Primary language is US English, additional languages Finnish, Swedish and German, Monthly update channel. No set up progress shown to user:

PHP Code:

```PHP
<Configuration> 
  <Info Description="This configuration file will install 32-bit O365 ProPlus. Excluded applications: OneDrive, SharePoint (Groove), Publisher and Skype for Business (Lync). " /> 
  <Add OfficeClientEdition="32" Channel="Monthly" AllowCdnFallback="TRUE" ForceUpgrade="TRUE"> 
    <Product ID="O365ProPlusRetail"> 
      <Language ID="en-us" /> 
      <Language ID="fi-fi" /> 
      <Language ID="de-de" /> 
      <Language ID="sv-se" /> 
      <ExcludeApp ID="Groove" /> 
      <ExcludeApp ID="OneDrive" /> 
      <ExcludeApp ID="Lync" /> 
      <ExcludeApp ID="Publisher" /> 
    </Product> 
  </Add> 
  <Property Name="SharedComputerLicensing" Value="0" /> 
  <Property Name="PinIconsToTaskbar" Value="TRUE" /> 
  <Property Name="SCLCacheOverride" Value="0" /> 
  <Updates Enabled="TRUE" /> 
  <RemoveMSI All="TRUE" /> 
  <AppSettings> 
    <Setup Name="Company" Value="Ten Forums" /> 
    <User Key="software\microsoft\office\16.0\common\toolbars" Name="screentipscheme" Value="0" Type="REG_DWORD" App="office16" Id="L_ShowScreenTips" /> 
  </AppSettings> 
</Configuration>
```

Another sample script to install 32-bit O365 Home in English, no additional languages. Update channel Semi-Annual, automatic updates turned off, setup progress shown to user and EULA automatically accepted. Office setup files will be downloaded to network share **\\W10PC\OfficeSetup** and Office installed from there:

PHP Code:

```php
<Configuration> 
  <Add OfficeClientEdition="32" Channel="Broad" SourcePath="\\W10PC\OfficeSetup" AllowCdnFallback="TRUE" ForceUpgrade="TRUE"> 
    <Product ID="O365HomePremRetail"> 
      <Language ID="en-us" /> 
   </Product> 
  </Add> 
  <Property Name="SharedComputerLicensing" Value="0" /> 
  <Property Name="PinIconsToTaskbar" Value="TRUE" /> 
  <Property Name="SCLCacheOverride" Value="0" /> 
  <Updates Enabled="FALSE" /> 
  <AppSettings> 
    <Setup Name="Company" Value="Ten Forums" /> 
  </AppSettings> 
  <Display Level="Full" AcceptEULA="TRUE" /> 
</Configuration>  
```

Notice that **Semi-Annual** channel is shown as *"Broad"* in script above. **Monthly Targeted** will show as *"Insiders*", **Monthly** as "*Monthly*", and **Semi-Annual Targeted** as *"Targeted"*.

![Custom install or change Microsoft Office with Office Deployment Tool](https://www.tenforums.com/IMAGES/OPTl.PNG) Part Three ![Custom install or change Microsoft Office with Office Deployment Tool](https://www.tenforums.com/IMAGES/OPTR.PNG)

## Configure and edit install options manually

  **3.1)** The configuration XML files can be created and edited manually. Sample files in **step 2.12** are a good base, showing some most common options. See full list of options and commands: [Configuration options for the Office Deployment Tool | Microsoft Docs](https://docs.microsoft.com/en-gb/DeployOffice/configuration-options-for-the-office-2016-deployment-tool)

**3.2)** Configuration file can be created in any text editor and named as you prefer. It must be saved as an XML file with extension **.xml**

**3.3)** See the second of sample script in **step 2.12**. As **Office Customization Tool** did not allow me to select **O365 Home** edition, I used **O365 ProPlus** instead when creating the script, therefore the Product ID when I exported the script to my PC was **<Product ID="O365ProPlusRetail">**. I changed this manually in Notepad to **<Product ID="O365HomePremRetail">**.

I also edited the script to exclude **Publisher**, change install source from network share to cloud (CDN), to change update channel to **Monthly**, and to add **Finnish** language pack.

This is the original configuration script I exported (downloaded) from **Office Customization Tool**:  

PHP Code:

```php
<Configuration> 
  <Add OfficeClientEdition="32" Channel="Broad" SourcePath="\\W10PC\OfficeSetup" AllowCdnFallback="TRUE" ForceUpgrade="TRUE"> 
    <Product ID="O365ProPlusRetail"> 
      <Language ID="en-us" /> 
   </Product> 
  </Add> 
  <Property Name="SharedComputerLicensing" Value="0" /> 
  <Property Name="PinIconsToTaskbar" Value="TRUE" /> 
  <Property Name="SCLCacheOverride" Value="0" /> 
  <Updates Enabled="FALSE" /> 
  <AppSettings> 
    <Setup Name="Company" Value="Ten Forums" /> 
  </AppSettings> 
  <Display Level="Full" AcceptEULA="TRUE" /> 
</Configuration>  
```

And here the same after my manual edits:

PHP Code:

```
<Configuration> 
  <Add OfficeClientEdition="32" Channel="Monthly" AllowCdnFallback="TRUE" ForceUpgrade="TRUE"> 
    <Product ID="O365HomePremRetail"> 
      <Language ID="en-us" /> 
      <Language ID="fi-fi" /> 
      <ExcludeApp ID="Publisher" /> 
   </Product> 
  </Add> 
  <Property Name="SharedComputerLicensing" Value="0" /> 
  <Property Name="PinIconsToTaskbar" Value="TRUE" /> 
  <Property Name="SCLCacheOverride" Value="0" /> 
  <Updates Enabled="FALSE" /> 
  <AppSettings> 
    <Setup Name="Company" Value="Ten Forums" /> 
  </AppSettings> 
  <Display Level="Full" AcceptEULA="TRUE" /> 
</Configuration> 
```

![Custom install or change Microsoft Office with Office Deployment Tool](https://www.tenforums.com/IMAGES/OPTl.PNG) Part Four ![Custom install or change Microsoft Office with Office Deployment Tool](https://www.tenforums.com/IMAGES/OPTR.PNG)

## Run Setup to install Office

**4.1**) Download **ODT**:

[![download](https://www.tenforums.com/images/dl3.png)](https://www.microsoft.com/en-us/download/details.aspx?id=49117)

**4.2)** Create a folder on root of C: drive, name it as **O365** (suggestion, you can name the folder as you prefer).

**4.3)** Unblock ([tutorial](https://www.tenforums.com/tutorials/5357-unblock-file-windows-10-a.html)) the downloaded EXE file, then run it. The file name is **officedeploymenttool_12345-67890.exe**, where **12345-67890** is current version number. At the moment of writing this, the ODT version is **11107-33602**.

**4.4)** Accept the license, click **Continue**:
[![Click image for larger version.   Name:	image.png  Views:	54  Size:	396.9 KB  ID:	216947](https://www.tenforums.com/attachments/tutorials/216947d1544501751t-custom-install-change-microsoft-office-office-deployment-tool-image.png)](https://www.tenforums.com/attachments/tutorials/216947d1544501751-custom-install-change-microsoft-office-office-deployment-tool-image.png)

**4.5)** To extract ODT, select the folder you created in **step 4.2** and click **OK**:
![Name:  image.png Views: 5768 Size:  74.8 KB](https://www.tenforums.com/attachments/tutorials/216948d1544501864-custom-install-change-microsoft-office-office-deployment-tool-image.png)

**4.6)** Copy your XML configuration file to same folder.

**4.7)** Open an **elevated Command Prompt**, move to **O365** folder with following command:

**cd \O365**

**4.8)** If you selected cloud (CDN) as source in **step 2.6**, enter following command:

**setup.exe /configure YourConfigurationFile.xml**![Name:  image.png Views: 5764 Size:  82.7 KB](https://www.tenforums.com/attachments/tutorials/216950d1544502725-custom-install-change-microsoft-office-office-deployment-tool-image.png)

**4.9)** Office will be installed as configured in configuration script.

**4.10)** If you selected a local folder or network share as **Source path** in **step 2.6**, Office setup files need to be downloaded first once with following command:

**setup.exe /download YourConfigurationFile.xml**

**4.11)** When downloaded, you can install Office from your local source with command in **step 4.8**

Notice that when installing from network share, ODT must be run on the computer on which you want to install Office. However, the configuration file and Source path for downloaded Office setup files can be located on a network share.

To download:

**setup.exe /download \\ServerName\O365Share\YourConfigurationFile.xml** 

To install:

**setup.exe /configure \\ServerName\O365Share\YourConfigurationFile.xml**



![Custom install or change Microsoft Office with Office Deployment Tool](https://www.tenforums.com/IMAGES/OPTl.PNG) Part Five ![Custom install or change Microsoft Office with Office Deployment Tool](https://www.tenforums.com/IMAGES/OPTR.PNG)

## Change existing Office installation

**5.1)** You can change your existing Office installation whenever you want to. You can create a new script, or edit existing as told in **step 3.35.2)** Run the modified script with **/configure** switch as shown in **step 4.8** or **step 4.11**, depending on if the configuration script is saved locally or on network share.

**5.3)** Office installation will be modified, features removed or added as told in configuration file.

That's it, geeks!

![Note](https://www.tenforums.com/images/notesmall10.png)   Note

Depending on selected options, using ODT to install or modify Office can be confusing because sometimes no indicator about progress is shown.

Even if you can see no progress, keep Command Prompt running until setup is ready, until you can see that Office Setup has finished running. Cursor shown under your command on column 1, setup is still running, either downloading or installing Office:
![Name:  image.png Views: 5738 Size:  46.1 KB](https://www.tenforums.com/attachments/tutorials/216953d1544506098-custom-install-change-microsoft-office-office-deployment-tool-image.png)

Setup ready, a prompt will be shown:
![Name:  image.png Views: 5739 Size:  40.4 KB](https://www.tenforums.com/attachments/tutorials/216954d1544506262-custom-install-change-microsoft-office-office-deployment-tool-image.png)