# How to contribute

The project main language is English.

## Commit message style

Commit messages should follow the [conventional commit specification](https://www.conventionalcommits.org/en/v1.0.0/).

The allowed structural elements are:
- `feat` for new features.
- `fix` for bug fixes.
- `chore` for changes to the build process or auxiliary tools and libraries such as documentation generation.
- `refactor` for code changes that neither fix a bug nor add a feature.
- `docs` for any documentation/README changes.

Commit messages should be structured in a way that can be read as if they were completing the sentence *"If applied, this commit will..."*. For example:

> feat: add new authentication method to API

Reads as *"If applied, this commit will add new authentication method to API"*.

## Branch naming

Branch names should be descriptive and use hyphens to separate words. They should also follow the same structure as commit messages, using the allowed structural elements. For example:
- `feat/add-user-authentication`
- `fix/issue-with-database-connection`
- `chore/update-dependencies`
- `refactor/improve-code-structure`
- `docs/update-contributing-guidelines`
