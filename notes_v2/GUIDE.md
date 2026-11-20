# Writing worked notes v2: instructions for the note writer

## What you are making

A small Vietnamese voice assistant (a 4B language model) answers the user's latest request by calling tools. Before its
first tool call it opens a thinking block and restates the request. We will train it to write one short **worked note**
right after that restatement. The note gives the reasoning that leads to the right calls. You write that note for each
training sample.

The note is written **at the first step of the final turn, before any tool runs in that turn**. It may use only what the
model can see at that moment:

- the current time and the list of coming days;
- the user profile;
- the earlier turns, including earlier tool calls and their results;
- the latest request.

It may **not** use anything that a tool call of this final turn returns. At test time the model has to write the note
without knowing what the reads will show, so the note must read **the same whatever the reads return**.

## The one big rule: state the rule, not the outcome

Whenever the right action depends on what a read of this turn returns, the note **names the read** and **states the
decision rule with both branches**. The wording is the same whether the real answer ends up acting, asking or stopping.
You are not shown the outcome for those samples. Do not try to guess it, and never write a note that sounds as if you
know it.

Forbidden in such notes (the checker rejects them):

- anything that reveals the branch: "không xác định được thì ...", "chỉ đọc", "không có nên thôi", "vẫn còn", "đã có",
  "đủ rồi", "chắc là", "có nhiều", "trùng tên", "không tìm thấy nên";
- a count or an ID that only the read shows ("3 đèn", "gói sub_2");
- committing to the write without the other branch ("tìm gói rồi huỷ").

Each sample has a `kind:` line. It tells you which shape to use. Several kinds can occur in one request: write each part
in its own shape, in the key's order.

| kind | shape (Vietnamese, adapt the words) |
|---|---|
| K0 explicit | no read decides anything. Resolved dates and times with the arithmetic, counts, references to earlier turns, the tools in order. |
| K1 pick one item | `<read> tìm <mô tả>; khớp đúng một thì <write tool>, không có hoặc nhiều thì sys.ask_missing_info cho <write tool>, params <param>.` |
| K2 condition on a read | `<read> xem <điều kiện>; <đúng> thì <tools>, không thì <thôi hoặc tools khác>.` |
| K3 select by a rule | `<read>; <write tool> từng mục <quy tắc chọn>.` Never the count and never the IDs. |
| K4 value lookup | `<read> lấy <giá trị>, rồi <write tool> với <giá trị đó hoặc công thức>.` An if/else on the value is a rule and is fine. |

## Your input

Read only the input file named in your task, for example `notes_v2/inputs/batch_000_p0.txt`. Do not open any other file
or folder: no data, no code, no other notes. Do not run scripts, and do not use the web or any other service. Each sample
block has these fields:

- `kind`: which shape above to use.
- `now` and `days`: the current date, time and weekday, and the coming days in D/M form.
- `profile` and `apps`: the user profile and the installed apps.
- `history`: earlier turns. Values that appear here are visible to the model.
- `request`: the latest request, which the model restates just before your note.
- `answer_type` (K0 and K4 only): `act` (some write), `ask` (ends by asking for a missing value), `read_only`, or
  `nothing` (no tool).
- `gold_calls`: the correct calls of the final turn.
  - For K0 and K4: all of them, in order.
  - For K1, K2 and K3: only the calls up to and including the **first read**. Then one line says that the rest depends
    on what the read returns. You are deliberately not shown the rest. For K1 only, a line names the `decision tool` and
    the `parameter the read must supply`.
- `tools` / `installed tools you may name after the read`: the tool descriptions.
  - For K2 and K3 this is the list of every installed tool. Choose the tools of the branches from the request, and name
    them: the "true" branch tool(s) and the "false" branch tool(s) if the request says what to do otherwise.
- `must_include`: strings that **must appear in your note exactly as written**:
  - dates as D/M (8/11);
  - times as H:MM in 24-hour form (9:45, 16:00);
  - numbers as plain digits (200000);
  - IDs verbatim (item_646);
  - tool and parameter names verbatim.
- `may_depend_on_reads`: values that may depend on a read made in this turn.
  - Write such a value only if the request or the history gives it directly, for example a spoken absolute time.
  - If it is relative to something a read returns ("lùi 30 phút", "trước đó một tiếng"), describe the relation and
    leave the value out.
  - The same holds for a value computed from what a read returns: a count, a sum, a price times a quantity, or a time
    minus a route duration. Write the rule, not the result, for example `tiền = 15 lít × giá đọc được`.

## What the note says

Keep only what helps the model get the calls right, in this order of importance:

1. **Dates and times resolved, with the arithmetic shown briefly.** For example:
   - `mười giờ kém mười lăm = 9:45`;
   - `T6 tuần sau = 13/11`;
   - `9:00 tới 9:45 = 45 phút`;
   - `trước 9:00 45 phút = 8:15`.
2. **The rule for anything a read decides** (see the table). Name the read, then the decision with both branches.
3. **Counts that the request or the history gives.** `2 lần`, `3 món`.
4. **References to earlier turns.** What "đó", "ổng", "bả", "bữa đó", "cái kia" or "món vừa đặt" points to in the
   history, with its ID when the history shows it.
5. **The tool choice.** Always name the first gold tool exactly (for example `calendar.list_events`), and the others in
   the key's order, every sub-task of a multi-step request. Add a reason of 2 to 5 words only when the choice is not
   obvious (an unusual app, or two apps that could both fit), taken from the tool's description.
6. **Asks that need no read.** Which required parameter is missing, then `sys.ask_missing_info` with the tool and the
   parameter names.
7. **Nothing.** For `answer_type: nothing`, say briefly why (no app can do it, small talk, the user takes it back, an
   invalid value, a condition that cannot be checked). The note must contain the exact words `không gọi tool`.
8. **Regional words.** Translate a Southern or Central word only when it matters for a value (`ni = nay`, `rứa = vậy`).

## No invented details

Every date, time, number and ID in the note must be one of these:

- a gold argument value of the final turn (K0 and K4: shown to you);
- a step of an explicit arithmetic chain that ends in such a value (`13:10 + 45 phút = 13:55`);
- (K1, K2, K3) the conversion of a value spoken in the request for the intended action (`năm mươi ngàn = 50000`).

In addition:

- Do not mention the current time or date unless it feeds such a chain.
- Do not mention optional parameters or filters the key does not set (`near`, `merchant`, `delay_minutes`, `label`, a
  workout `time`, ...). In K1, K2 and K3 notes, the only parameter name you may write is the one you were given (the
  decision parameter); describe the rest in words.
- Write search terms exactly as the key's query. Do not add words or numbers ("xăng", not "xăng 92"). In K1, K2 and K3
  you are not shown the later queries: take them from the request.
- When a tool's name could suggest a different option value than the key uses, write the key's value
  (`email.archive` with `action delete`). In K1, K2 and K3, do not write option values at all unless the request names
  them.

## Style and limits

- Write Vietnamese, terse and telegraphic. Arrows (→), `=`, `+`, `;` and parentheses are fine.
- Aim for **10 to 35 words**. The hard limits are 60 words and 100 tokens. Shorter is better.
- Use one line only. Do not use the characters `<` `>` `{` `}`, the word "think", or JSON.
- Do not restate the request, since the model already did that just before the note. Do not copy the call JSON. Do not
  mention default values.
- Name only installed tools (the ones in your input) and `sys.ask_missing_info`.

## Examples (made up, not from the data)

**K0 explicit**

| | Note |
|---|---|
| good (gold `calendar.update_event` event_id evt_12, start 8/11 15:00, end 8/11 16:00; the history shows evt_12 "Họp chị Mai" on 8/11 at 9:00) | `Họp chị Mai mai 8/11 là evt_12 (lượt trước). 3 giờ chiều = 15:00, giữ 1 tiếng → 16:00. calendar.update_event.` |
| good (nothing) | `Người dùng rút lại yêu cầu nên không gọi tool.` |
| bad (invents the current time) | `Bây giờ 14:05, 3 giờ chiều = 15:00. calendar.update_event.` |

**K1 pick one item**

| | Note |
|---|---|
| good | `contacts.search Lan; khớp đúng một người thì sms.send, không có hoặc nhiều người thì sys.ask_missing_info cho sms.send, params contact_id.` |
| bad (outcome leak) | `contacts.search Lan, có hai Lan nên không xác định được thì sys.ask_missing_info.` |
| bad (commits to the write) | `contacts.search Lan lấy contact_id rồi sms.send.` |
| bad (invents a filter) | `subscriptions.list lọc theo merchant, khớp thì subscriptions.cancel.` |

**K2 condition on a read**

| | Note |
|---|---|
| good (request: "mai mưa thì huỷ lịch đá bóng, không thì thôi") | `weather.get_forecast mai 12/11 xem có mưa không; mưa thì calendar.list_events tìm buổi đá bóng rồi calendar.cancel_event, không mưa thì thôi.` |
| good (request: "ví đủ thì trả bằng ví, không thì bằng ngân hàng") | `ewallet.get_balance xem ví đủ tiền hoá đơn không; đủ thì bill_payment.pay trả bằng ví, không đủ thì bill_payment.pay trả bằng ngân hàng.` |
| bad (states the outcome) | `weather.get_forecast, mai không mưa nên chỉ đọc.` |
| bad (one branch only) | `weather.get_forecast, mưa thì calendar.cancel_event.` |

**K3 select by a rule**

| | Note |
|---|---|
| good (request: "tắt hết đèn tầng 2, phòng nào đang chạy quạt thì để đèn") | `smart_home.get_status tầng 2; smart_home.light_set tắt từng đèn, bỏ phòng đang chạy quạt.` |
| bad (count and IDs) | `smart_home.get_status; tắt 3 đèn light_01, light_02, light_05.` |

**K4 value lookup**

| | Note |
|---|---|
| good (gold: `banking.get_balance`, then `banking.transfer` from_account_id acc_01, amount 2000000) | `banking.get_balance xem tài khoản nào đủ 2000000; banking.transfer 2000000 từ tài khoản đó.` |
| bad (invents) | `banking.get_balance, số dư 5 triệu, chuyển 2000000 từ acc_01 có ghi chú góp vàng.` |

## Output

For each input part, write one JSONL file to the path named in your task, for example
`notes_v2/parts/batch_000_p0.jsonl`. Put one line per sample, in input order, exactly `{"id": "train_XXXX", "note": "..."}`,
and use valid JSON. Write the whole file with the Write tool. When asked to fix notes, write only the listed ids to the
file path you are given.
