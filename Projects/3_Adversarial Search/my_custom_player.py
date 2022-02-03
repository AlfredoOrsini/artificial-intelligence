
from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
        if state.ply_count<=2:
            self.queue.put(random.choice(state.actions()))
        else:
            i=1
            while True:
                self.queue.put(self.negamax(state, i))
                i+=1

    def negamax(self, state, depth):
        '''
        Parameters
        ----------
        state : Isolation
            The state of an Isolation gameboard.
        depth : integer
            the max depth that we are willing to search.

        Returns
        -------
        An action
            The action with the max value (searching at that depth).

        '''

        def nega_max(state, a, b, depth, player):
            if state.terminal_test() : return state.utility(self.player_id) * player

            if depth <= 0 : return self.score(state) * player

            value = float('-inf')
            for action in sorted(state.actions(), key=lambda x: self.score(state.result(x)), reverse=player==1):
                value = max(value, -nega_max(state.result(action),-b, -a, depth-1, -player))
                a=max(a,value)
                if a>=b: return value
            return value

        return max(state.actions(), key=lambda x: -nega_max(state.result(x), float('-inf'), float('inf'), depth - 1, -1))

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
