---
title:  'Data Management FAQs'
sidebar_position: 1
---

# Z by HP AI Studio Data Management FAQs	 
Last Updated: April 2025

This document answers the commonly asked questions related to AI Studio, data collection, transmission, storage, retention, and disposal of data.

## AI Studio

> **What is AI Studio?**

Z by HP AI Studio is an enterprise-grade SaaS platform designed for collaboration in local compute environments. With AI Studio users can create an account, download and install the app on Windows or Linux and invite collaborators to start creating AI projects locally and accelerate AI model development. 

> **How are updates provided to AI Studio?**

Users are made aware of updates to AI Studio by notice provided through the application. Users can then elect to update/install new versions of the software. New features may require that all collaborators update to the latest version to take advantage of features. To update, users will need to visit ‘Account’ > ‘General’ > ‘App Version’ to review release notes and update the software.  Updates are also posted on ZDocs and on the Z by HP Data Science and AI Community. 

## Cloud Technology/Data Centers

> **What cloud technology and data centers are leveraged by AI Studio?**

AI Studio services are hosted on Amazon Web Services (AWS).  All data associated with a given account is hosted in a single data center. 

To learn more, visit [AWS.](https://aws.amazon.com.)


AI Studio persists product and usage data in MongoDB.

AI Studio uses HP-ID as an Identity Provider (IdP).  HP-ID performs user authentication and provides the results of authentication to AI Studio in the form of JWTs (JSON Web Tokens).  HP-ID is hosted on AWS.

> **How does the data flow throughout the AI Studio activation process?**

Account owners follow a link received at their designated owner email address. The welcome email includes instructions to download and install the AI Studio application. Account owners onboard additional users by inviting them to the account from within the application.  Users onboard in a similar manner.  Account owners and users are prompted to create an HP-ID account associated with their e-mail address if they have not done so previously.  After logging in, account data is synchronized across the devices of all users according to their role defined within the application by the account owner.

> **How does data flow between devices and various Z by HP AI Studio components and integrated software solutions?**

Data created through the use of Z by HP AI Studio (such as notebooks, ML Flow experiments and tensorboard logs) is first stored locally on the device where it originates. Users can share files with other users of the account by placing them in a project’s “shared” folder. AI Studio establishes an end-to-end encrypted peer-to-peer connection between the account users to sync data. If a peer-to-peer connection is not possible due to network restrictions, Z by HP AI Studio will use a relay hosted in the cloud to facilitate the transfer between peers. Cloud relays (and by association HP itself) cannot decrypt the data as it traverses the relay because the associated keys are generated and stored only within the local filesystem of each peer.

AI Studio offers functionality that allows users to integrate data stored in various cloud storage solutions.  Access to that data is controlled by existing mechanisms established within the user’s IT organization.  When AI Studio accesses the data, it is sent directly to or from the device and then to or from the cloud provider using the cloud provider's software libraries. 

> **Can the customer influence in which region/data center their data is processed or sent to?**

No.

## Data Collection

> **What data does AI Studio collect and how is it used?**

| Purpose of Data Collection  | Data Collected |Description of Collected Data |
| :------------- |:-------------|:-------------|
| Account management     | Account data     |Information such as how a customer purchases or signs up for AI Studio services, support history with respect to incidents submitted via AI Studio, customer D-U-N-S number, and data related to the AI Studio account used to perform transactional services like account management.|
| Ensure AI Studio and services work properly      | Application data     |Software version and installation status of AI Studio software application. Data sharing choices and preferences of AI Studio customers.|
| Enable account setup, identity management, entitlement validation, marketing      | Contact data     |Personal and/or business contact data, including first name, last name, mailing address, telephone number, fax number, e-mail address that will be used for AI Studio customer account setup and validation, service entitlement, and e-mail notifications around product improvements, incidents and services, and e-mail with promotions and campaigns for HP products.|
| Deliver remote capabilities and support for incidents and in-app reports/dashboards.      | Device data     |Hardware components such as CPU and storage. Graphics card utilization, including overall utilization of cores and memory, clocks, temperature, and power state. Usage includes both processing and VRAM. Network utilization: network transmission rates from the device.Note: This data collection is turned on by default for all customers and an option is provided in AI Studio for customers to enable or disable data collection of this information.|
|Ensure AI Studio is providing best-in-class value to customers|Feature usage data|Which features in the app are used, who used them, and frequency of use. This data is collected by default to deliver the product insight feature.|
|Deliver proactive service maintenance and management of the application by pinpointing the location of the device|Location data|Geolocation data to enable location-based services. This data enables compliance with local laws and regulations and performance optimization of the application.|
|Authentication andnauthorization of user access to HPID accounts and associated services|Security credentials|HP AIS authentication and authorization complies with HP policies.  AI Studio does not directly store user passwords.  It stores and manages tokens derived from the authentication process that are used to authorize requests to cloud services.|
|Responses to User Support Requests|Entitlement data |A support ticket system is integrated within the AI Studio platform for users to send questions to and share information about issues with HP Support. HP’s systems collect and save the associated responses. In addition to account, entitlement, and contact data, the type of data collected is dependent on the questions asked.
|Responses to Community Engagement| Contact data| An in-app link to a community forum for engagement is integrated within the AI Studio platform. Through the community, users can send questions to and share information about their experiences with peers and HP community managers. The community system will collect and save the posts, issues, suggestions, and responses shared by users. The type of data collected is dependent on information shared.|

> **What types of data are not collected by AI Studio?**

AI Studio does not collect the following types of data:
- Demographic information (with the exception of country or language preferences).
- Financial account information, credit or debit card numbers, credit records, or payment data.
- Social media account information.
- Government-issued identifier such as social security, social insurance number, or Government ID.
- Sensitive data such as ethnic origin, political beliefs, trade union membership, health data, sexual orientation, and genetic data.
- Health information.

> **Can customers access their telemetry data that AI Studio transfers to the Z BY HP Backend?**

Telemetry data is anonymized and aggregated before storage and thus cannot be traced back to its origin after collection.  However, customers do get visibility in the app for account insights and delivery of analytics and features.

> **How often does AI Studio submit telemetry data to cloud services for storage?**

Telemetry data is submitted on demand based on usage.

> **How much data is generated and collected by AI Studio and sent to cloud services?**

Data is collected at the moment of user interaction. If a network connection is not available, data is stored locally for submission until a network connection is available.

## Security

> **What security measures are used by AI Studio to protect personal data?**

When capturing, transmitting, and storing data, AI Studio uses a variety of security technologies and processes to help protect your personal data from unauthorized access, use, or disclosure. These include:
- data in transit is protected using TLS 1.2 or better
- data at rest is protected using AES 256
- RBAC is used to control access to data
- only users with a business need are granted access to customer data
- periodic reviews are performed to ensure RBAC access is up to date
- AIS uses JWT tokens derived from the authentication process in all requests to backend cloud services.
- when applicable, data is anonymized by replacing sensitive information with a suitable proxy replacement (e.g. SHA256 or one-way hash) of it before storage.

> **What type of security testing is performed on AI Studio?**

AI Studio employs various security tests including but not limited to:
-	Penetration testing of all public facing cloud services
-	Static code analysis of all HP source code
-	Container scanning of all HP built images
AI Studio undergoes these tests for all new functionalities being released and periodically for all minor enhancements to existing functionality.

> **What are the processes and procedures to ensure physical and environmental security?**

Most AI Studio cloud services run on Amazon Web Services. Amazon Web Services is responsible for physical and environmental security of its data centers.
The Amazon Web Services data centers are Tier 3+ as per [AWS documentation.](https://aws.amazon.com/compliance/uptimeinstitute/)
- The Amazon Web Services compliance and certifications can be found at [AWS Compliance.](https://aws.amazon.com/compliance/)
- Amazon Web Services takes care of protection against external and environmental threats (fire, flood, earthquakes etc.).

AI Studio uses tools from Amazon Web Services (e.g., Cloud Watch) to monitor the performance of Amazon Web Services.

>**What authentication and authorization methods does Z by HP AI Studio implement?**

AIS implements authentication and authorization of users using HP-ID. Once authenticated, AIS uses a role-based access control (RBAC) system to authorize requests made on behalf of users throughout the platform.  The RBAC system ensures users may only access the data they are authorized to according to their role.  AIS uses TLS 1.2 and 1.3 to protect data in transit and AES 256 encryption to protect data at rest.


## Data Transmission and Storage

>**What data can a customer opt-out from sharing with AI Studio?**

The following table provides opt-out information:

The application requires specific data to function and deliver application features, including account management and user account analytics shared with the user. Customers are provided the option to consent to terms prior to gaining access to the software. Customers who do not consent will be unable to log in or access features within the application.

|Data Category |	Opt-out |	Additional Comments|
| :------------- |:-------------|:-------------|
|Account data|	No	|
|Application data|	No	
|Contact data|	No|	A customer's first name, last name, e-mail address, and company/account name, and country information are mandatory information for a customer to provide for log in and entitlement validation.|
|Device Data; Feature usage Data; Location Data; Entitlement Data| No |Device data is required for AI Studio to deliver its functionalities. To deliver functional requirements and support application regional compliance, device, feature, location, and entitlement data are mandatory information for a customer to provide.|
|Security Credentials| No |AI Studio stores and uses tokens for authorization to cloud services.|

>**How is the data transmission to Z BY HP secured?**

AI Studio leverages TLS 1.2 or better to transmit data between the device and the U.S. data centers, the U.S. Analytics data center, and the U.S. Identity Management data center.

> **What ports need to be considered for customer firewalls?**

No inbound ports are required.  AIS uses port 443 for connections to cloud services.  Locally, AIS uses ephemeral ports (49152-65535) for inter-process communication.

>**Is data in databases stored securely?**

All databases in the cloud are encrypted at rest using AWS KMS with keys created, rotated and managed by HP.  Local databases are encrypted with AES256 using a locally generated key that is stored in the platform credential manager.  Windows credential manager and the gnome key ring for Windows and Linux respectively.

>**Is data securely disposed of when no longer needed?**

Analytics data is deleted permanently two years after its creation.  Account data is deleted 60 days after the expiration of the account subscription.

:::note

For data protection purposes, all personal data is removed prior to transmission to and storage.

:::

>**How is access to data restricted?**

HP controls access to analytics data using RBAC policies.  Periodic reviews are performed to ensure that users with access have a business need.  Users that do have access are required to use 2 factor authentication whenever accessing the data.

>**Are the data anonymized?**

Analytics data is stripped of personally identifiable information (PII) before transmission to and storage in the analytics datacenter by replacing PII (e.g. e-mail address, first and last names) with proxy information (e.g. one way hash of the input data).

>**Does AI Studio share its data with its Z BY HP suppliers, and if so, does it include personal information?**

Yes, AI Studio shares account, application, contact, and device data with some of Z BY HP's key suppliers to ensure the correct functionality of all AI Studio features. HP Suppliers are contractually bound to by HP guidelines and requirements with regards to collection, use and maintenance of collected data.

>**Does AI Studio provide separate, dedicated database(s) for customers' exclusive use of their data?**

No.

>**Is customer data and information co-hosted with the data from other organizations or companies on the same physical server(s)?**

Yes.

>**What are the sources and methodology of data capture and frequency of transmission by Z by HP AI Studio?**

The sources and methodology of data capture and frequency of data transmission include:

|Data Category | Source |  Methodology| Transmission Frequency |
| :------------- |:-------------|:-------------|  :-------------|
|Account data |	HP order management system and cloud-based portal | The data is provided by customers and is entered manually by an HP. Service Expert into HP’s order management system. Confirmations and authentications are automated and prompt the customer to complete account set-up including creating an HPID account. | In real-time, based on changes to account data.|
|Application data |	AI Studio cloud-based data sync and event telemetry | Automatic |	Based on changes to application data, changes sync across accounts every minute.|
|Contact data |	AI Studio Entitlement Portal | Collected during account creation either as a trial or paid customer process (see Account data)|	In real-time, based on changes to account data.|
|Device data | AI Studio application |	Automatic | Device data used for analytics including settings, hardware usage, and status, are collected and transmitted depending upon the network connection status.|
|Feature usage data	| AI Studio application |	Automatic |	Event data for analytics is transmitted in real-time based on events triggered. |
|Location data | AI Studio application | Automatic	| In real-time or once every 12 hours, based on changes to geo-location data. |
|Entitlement Data |	HP order management system | Automatically created during the account creation process (see Account data) |
|Security credentials | Cloud-based HPID web application | The data is entered by the customer manually in the cloud-based HPID portal. AI Studio relies on external applications for security credential authorization and authentication and does not store user credential data.	| In real-time, based on changes to security credentials data.|

## General Data Protection Regulation (GDPR)

>**What is GDPR compliance?**

The General Data Protection Regulation (GDPR) is a European-wide regulation for the protection of European citizens' data that was established in May 2018 and defined rules related to the protection of natural persons regarding the processing of personal data and rules relating to the free movement of personal data (Ref: https://gdpr-info.eu/art-1-gdpr/). Currently there is no ID or license required or available for GDPR.

>**What is the Z BY HP and AI Studio approach to GDPR?**

Z BY HP has a long-standing history of industry leadership in privacy and data protection. Together with our robust portfolio of products and services, we can support our customers' and partners' efforts in protecting personal data. With respect to AI Studio, Z BY HP acts as a Data Processor. Please refer to Data Processor section on [Z BY HP Privacy Central.](https://www.hp.com/us-en/privacy/privacy-central.html) As a global company, it is possible that any information you provide may be transferred to or accessed by Z BY HP entities worldwide in accordance with the Z BY HP Privacy Statement and based on the International Privacy Programs listed in the International Data Transfers section.

> **Do you have an assigned Data Protection & Privacy Officer or equivalent?**

Yes. For more information refer to the Frequently Asked Questions (FAQ) in the Data Processor section on [Z BY HP Privacy Central.](https://www.hp.com/us-en/privacy/privacy-central.html)
