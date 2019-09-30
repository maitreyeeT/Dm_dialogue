import re


class Player(object):

  def __init__(self, rules):
    self.rules = rules

  def __call__(self, *args):
    blackboard = args[0]
    self.take_turn(blackboard)

  def take_turn(self, blackboard):

    curr = blackboard['current']

    non_term = [x for x in re.finditer('[A-Z]', curr)]

    if len(non_term) == 0:
      return False

    for i, nt in enumerate(non_term):

      # nt is a match object
      sym = nt.group(0)
      start, end = nt.start(0), nt.end(0)

      if sym in self.rules.keys():
        ## compute
        exp = self.rules[sym]
        exp = exp.split('|')
        if len(exp) > 1:
          exp = self.rule_choice(sym, exp)
        else:
          exp = exp[0]

        # Now exp is the chosen expansion

        blackboard['current'] = '{}{}{}'.format(curr[:start], exp, curr[end:])

        return True

    return False

  def rule_choice(self, np, exp):
    return exp[0]


end_state = lambda x: len(re.findall('[A-Z]', x['current'])) == 0


def player_turn(player, blckbrd, k):
  n = 1
  while player(blckbrd) and n <= k:
    n += 1
    if end_state(blckbrd):
      return False

  return True


if __name__ == "__main__":
  # default dict
  blckbrd = {}

  # define the protocol
  k = 1

  blckbrd['current'] = 'S'

  p_1 = {
    'S': 'G Q',
    'G': 'hello|hi',
    'Q': 'how are you A'
  }

  p_2 = {
    'A': 'fine'
  }

  player_1 = Player(p_1)
  player_2 = Player(p_2)

  while not end_state(blckbrd):

    # Player 1 turn
    print('Player 1 turn')
    if not player_turn(player_1, blckbrd, k):
      break

    print(blckbrd)

    # Player 2 turn
    print('Player 2 turn')
    if not player_turn(player_2, blckbrd, k):
      break
    print(blckbrd)
