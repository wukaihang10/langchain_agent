from __future__ import annotations

import re


class CodeTokenizer:
    """
    面向源代码和代码查询的基础 Tokenizer。

    它会同时保留：

    1. 完整标识符：
       _finalize_tool_call

    2. snake_case 子词：
       finalize
       tool
       call

    3. CamelCase / PascalCase 子词：
       RepositoryState
       repository
       state

    4. 中文连续文本和中文二元词组。
    """

    _TOKEN_PATTERN = re.compile(
        r"[A-Za-z_][A-Za-z0-9_]*" r"|\d+(?:\.\d+)*" r"|[\u4e00-\u9fff]+"
    )

    _CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])" r"|(?<=[A-Z])(?=[A-Z][a-z])")

    def tokenize(
        self,
        text: str,
    ) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        tokens: list[str] = []

        for match in self._TOKEN_PATTERN.finditer(text):
            raw_token = match.group(0)

            tokens.extend(self._expand_token(raw_token))

        return tokens

    def _expand_token(
        self,
        raw_token: str,
    ) -> list[str]:
        if self._is_chinese_token(raw_token):
            return self._expand_chinese_token(raw_token)

        if raw_token[0].isdigit():
            return [raw_token.casefold()]

        candidates = [
            raw_token.casefold(),
        ]

        for snake_part in re.split(
            r"_+",
            raw_token,
        ):
            if not snake_part:
                continue

            camel_parts = self._CAMEL_BOUNDARY.split(snake_part)

            candidates.extend(part.casefold() for part in camel_parts if part)

        return self._unique_in_order(candidates)

    @staticmethod
    def _is_chinese_token(
        token: str,
    ) -> bool:
        return all("\u4e00" <= character <= "\u9fff" for character in token)

    @staticmethod
    def _expand_chinese_token(
        token: str,
    ) -> list[str]:
        candidates = [token]

        if len(token) > 1:
            candidates.extend(
                token[index : index + 2] for index in range(len(token) - 1)
            )

        return CodeTokenizer._unique_in_order(candidates)

    @staticmethod
    def _unique_in_order(
        values: list[str],
    ) -> list[str]:
        seen: set[str] = set()
        unique_values: list[str] = []

        for value in values:
            if not value:
                continue

            if value in seen:
                continue

            seen.add(value)
            unique_values.append(value)

        return unique_values
