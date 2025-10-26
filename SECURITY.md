# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within ezpz, please send an email to saforem2@gmail.com. All security vulnerabilities will be promptly addressed.

Please do not publicly disclose the vulnerability until it has been addressed.

## Security Best Practices

### For Users

1. **Keep Dependencies Updated**: Regularly update ezpz and its dependencies
2. **Review Dependencies**: Check the dependencies before installing
3. **Use Virtual Environments**: Always use virtual environments to isolate dependencies
4. **Validate Inputs**: When using ezpz in your applications, validate all inputs

### For Developers

1. **Code Review**: All code changes must be reviewed before merging
2. **Dependency Management**: Regularly audit dependencies for known vulnerabilities
3. **Secure Coding Practices**: Follow secure coding practices
4. **Environment Variables**: Never commit sensitive information like API keys or passwords
5. **Input Validation**: Validate all user inputs and external data
6. **Error Handling**: Implement proper error handling without exposing sensitive information

## Security Tools

We use the following tools to maintain security:

- **Bandit**: Security linter for Python code
- **Safety**: Checks dependencies for known security vulnerabilities
- **GitHub Security**: Automated security scanning

## Security Testing

Security testing is performed as part of our CI/CD pipeline using Bandit. Results are reviewed regularly.

## Incident Response

In the event of a security incident:

1. **Containment**: Immediately contain the vulnerability
2. **Investigation**: Investigate the scope and impact
3. **Remediation**: Develop and deploy a fix
4. **Communication**: Notify affected parties
5. **Post-mortem**: Document lessons learned

## Contact

For security-related questions or concerns, please contact:

- Sam Foreman (Maintainer): saforem2@gmail.com