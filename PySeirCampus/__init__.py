'''
Copyright (c) 2020 Matthew Zalesak

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import ast
from collections import defaultdict
import copy
from datetime import datetime, timedelta
from enum import Enum
from math import ceil
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import re

# Define functions to convert a dateframe into a semester object.

class MeetType(Enum):
    COURSE = 1
    SOCIAL = 2
    UNTRACEABLE = 3

weekday_lookup = {
    'M' : 0,
    'T' : 1,
    'W' : 2,
    'R' : 3,
    'F' : 4,
    'S' : 5,
    'U' : 6
}

class Semester:
    def __init__(self, datafile, holidays = set()):
        self.meetings = {}
        self.students = {}
        self.groups = defaultdict(set)
        self.student_enrollment = defaultdict(set) # TODO : Load this!
        self.meeting_enrollment = defaultdict(set)
        self.meeting_dates = defaultdict(dict)
        self.meeting_rooms = {} # Course --> Student --> Student --> Weight
        self.meeting_type = {}
        
        with open(datafile) as fin:
            for i, line in enumerate(fin):
                try:
                    data = ast.literal_eval(line)
                except:
                    print(i)
                    print(line)
                    print('There was an error reading your data.')
                    raise
                if 'student' in data:
                    data = data['student']
                    self.students[data['id']] = data['demographics']
                elif 'meeting' in data:
                    data = data['meeting']
                    meeting = data['id']
                    info = data['info']
                    meets_raw = data['meets']
                    meets = []
                    for meet in meets_raw:
                        meet = list(meet)
                        meet[0] = datetime.strptime(meet[0], '%m/%d/%Y')
                        meet[1] = datetime.strptime(meet[1], '%m/%d/%Y')
                        meets.append(meet)
                    members = set(data['members'])
                    self.add_meeting(meeting, info, meets, members, holidays = holidays)
                elif 'group' in data:
                    data = data['group']
                    group = data['name']
                    members = set(data['members'])
                    self.groups[group] = members
                else:
                    raise ValueError('There was an unknown label in the data.')
    
    def add_meeting(self, name, info, meets, members, meet_type = MeetType.COURSE,
                    holidays = set()):
        self.meetings[name] = info
        self.meeting_type[name] = meet_type
        self.meeting_enrollment[name] = members
        for member in self.meeting_enrollment[name]:
            self.student_enrollment[member].add(name)
        if type(meets) == list:
            for meet in meets:
                date, end, weekday, duration = meet
                weekday = set([weekday_lookup[x] for x in weekday])
                while date <= end:
                    if date.weekday() in weekday and date not in holidays:
                        self.meeting_dates[date][name] = duration
                    date += timedelta(days = 1)
        else:
            for date, duration in meets.items():
                self.meeting_dates[date][name] = duration
    
    def remove_meeting(self, name):
        del self.meetings[name]
        for member in self.meeting_enrollment[name]:
            self.student_enrollment[member].discard(name)
        del self.meeting_enrollment[name]
        del self.meeting_type[name]
        date = min(self.meeting_dates)
        end_date = max(self.meeting_dates)
        while date <= end_date:
            if name in self.meeting_dates[date]:
                del self.meeting_dates[date][name]
            date += timedelta(days = 1)
    
    def clean_student_list(self):
        group_members = set()
        for members in self.groups.values():
            group_members.update(members)
        active_students = set()
        for s in self.students:
            if len(self.student_enrollment[s]):
                active_students.add(s)
        to_keep = group_members.union(active_students)
        to_remove = [s for s in self.students if s not in to_keep]
        for s in to_remove:
            del self.students[s]
            del self.student_enrollment[s]
    
    def summarize(self):
        print('Semester object with', len(self.students), 'students,',
              len(self.meetings), 'courses, and', len(self.groups), 'groups.')
        course_sizes = np.array([len(ss) for ss in self.meeting_enrollment.values()])
        print('Course sizes: min', min(course_sizes), 
              ', avg', round(np.mean(course_sizes), 1),
              ', max', max(course_sizes))
        course_durations = defaultdict(list)
        date = min(self.meeting_dates)
        end_date = max(self.meeting_dates)
        while date <= end_date:
            for course, duration in self.meeting_dates[date].items():
                course_durations[course].append(duration)
            date += timedelta(days = 1)
        durations = []
        for course in course_durations:
            durations.append(np.mean(course_durations[course]))
        durations = np.array(durations)
        print('Course durations: min', min(durations), 
              ', avg', round(np.mean(durations), 1),
              ', max', max(durations))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
        if max(course_sizes) > 100:
            ax1.hist(course_sizes, 
                     bins = range(0, 20 * ceil(max(course_sizes) / 20) + 20, 20))
        else:
            ax1.hist(course_sizes)
        ax1.set_title('Distribution of Course Sizes')
        ax1.set_xlabel('Number of Students')
        ax1.set_ylabel('Number of Occurances')
        ax2.hist(durations)
        ax2.set_title('Distribution of Course Durations')
        ax2.set_xlabel('Average length per class')
        ax2.set_ylabel('Number of Occurances')
        plt.show()

class BasicInfectionDuration:
    def __init__(self, rate_contagious, rate_recovery):
        self.rate_contagious = rate_contagious
        self.rate_recovery = rate_recovery
    def duration(self, simulation, student, date):
        contagious = date + timedelta(days = np.random.geometric(p = self.rate_contagious))
        recovered = contagious + timedelta(np.random.geometric(p = self.rate_recovery))
        return contagious, recovered, False

class VariedResponse:
    ''' rate_contagious - daily rate at which exposed person beomces infectious.
        rate_recovery - daily rate at which infectious, asymptomatic person recovers.
        rate_symptoms - daily rate at which infectious person develops symptoms.
        a_ratio - fraction of people who will be asymptomatic.
    '''
    def __init__(self, rate_contagious, rate_recovery, rate_symptoms, a_ratio):
        self.rate_contagious = rate_contagious
        self.rate_recovery = rate_recovery
        self.rate_symptoms = rate_symptoms
        self.a_ratio = a_ratio
    def __str__(self):
        def nice_look(num):
            num = str(num)
            if len(num) > 6:
                return num[:6]
            else:
                return num
        return ''.join(['VariedResponse(contagious: ', nice_look(self.rate_contagious),
                                 ', recovery: ', nice_look(self.rate_recovery),
                                 ', symptoms: ', nice_look(self.rate_symptoms),
                                 ', asympt ratio: ', nice_look(self.a_ratio), ')'])
    def duration(self, simulation, student, date):
        until_contagious = np.random.geometric(p = self.rate_contagious)
        symptomatic = np.random.uniform() > self.a_ratio
        if symptomatic:
            sick_time = np.random.geometric(p = self.rate_symptoms)
        else:
            sick_time = np.random.geometric(p = self.rate_recovery)
        until_removed = until_contagious + sick_time  # Relative first date person is recovered.
        contagious_d = date + timedelta(days = until_contagious)
        remove_d = date + timedelta(days = until_removed)
        return contagious_d, remove_d, symptomatic

def DefaultInfectionDuration():
    return VariedResponse(1 / 3.5, 1 / 4.5, 1 / 2, 0.75)

class DefaultInterventionPolicy:
    def testing(self, simulation, date):
        pass
    def __str__(self):
        return 'DefaultInterventionPolicy(no intervention)'

class DefaultContactTracing():
    def trace(self, simulation, date):
        pass
    def __str__(self):
        return 'DefaultContactTracing(no action)'

class Parameters:
    ''' This object holds default parameter settings.
        Modify each item that you want non-default values for. '''
    def __init__(self, reps = 10):
        self.verbose = False # For debugging.
        self.rate = 1 / (65.6 * 60 * 7)
        self.daily_spontaneous_prob = 0.0001 # 10 out 14500 exposed externally per week.
        self.contact_tracing = DefaultContactTracing()
        self.intervention_policy = DefaultInterventionPolicy()
        self.infection_duration = DefaultInfectionDuration()
        self.quarantine_length = 14
        self.preclass_interaction_time = 5
        self.initial_exposure = 10
        self.preprocess = lambda x : x # Default is no preprocessing.
        self.repetitions = reps
        self.start_date = datetime(2020, 9, 2)
        self.end_date = datetime(2020, 11, 13)
    def info(self):
        for x in self.__dir__():
            if x[0] != '_' and x != 'info':
                if type(self.__getattribute__(x)) == float:
                    print(x, '=', round(self.__getattribute__(x), 5))
                else:
                    print(x, '=', self.__getattribute__(x))

# Actual simulator.

class Statistics:
    def __init__(self):
        self.Tn = []
        self.Qn = []
        self.Sn = []
        self.En = []
        self.In = []
        self.Rn = []

class State:
    def __init__(self):
        self.S = set()
        self.Sc = defaultdict(set) # Susceptible people by course
        self.E = {}
        self.Ia = {}
        self.Is = {}
        self.Q = {}
        self.Qe = {}
        self.Qa = {}
        self.Qs = {}
        self.R = {}

class Simulation:
    def __init__(self, semester, parameters):
        self.semester = semester
        self.parameters = parameters
        self.date = parameters.start_date
        self.Data = {} # This records information about infected people.
        self.state = State()
        self.contact_trace_request = defaultdict(set)
        self.test_requests = defaultdict(set)
        self.positive_tests = defaultdict(set)
        self.reinsert = defaultdict(set)
        self.statistics = Statistics()
        for s in semester.students:
            self.state.S.add(s)
            for c in semester.student_enrollment[s]:
                self.state.Sc[c].add(s)
    
    def daily_update(self):
        ''' Update.'''
        
        # Process incoming test results.
        for s in self.positive_tests[self.date]:
            self.initiate_quarantine(s, self.date, positive_test = True)
        
        # This covers the general case of people waiting to recover.
        def pool_recover(pool, report = False):
            transitions = set()
            for s, d in pool.items():
                if d == self.date:
                    transitions.add(s)
            for s in transitions:
                del pool[s]
                self.state.R[s] = self.date
                if report:
                    self.contact_trace_request[self.date].add(s)
        
        pool_recover(self.state.Qe)
        pool_recover(self.state.Qa)
        pool_recover(self.state.Qs)
        pool_recover(self.state.Ia)
        pool_recover(self.state.Is, report = True)
        
        # This covers the special case of people who were not infected.
        Q_transitions = set()
        for s, d in self.state.Q.items():
            if d == self.date:
                Q_transitions.add(s)
        for s in Q_transitions:
            del self.state.Q[s]
            self.state.S.add(s)
            for c in self.semester.student_enrollment[s]:
                self.state.Sc[c].add(s)
        
        # This covers the special case of people who are now becoming infectious!
        Ia_transitions = {}
        Is_transitions = {}
        for s, (i, r, symptomatic) in self.state.E.items():
            if i == self.date:
                if symptomatic:
                    Is_transitions[s] = r
                else:
                    Ia_transitions[s] = r
        for s, r in Ia_transitions.items():
            del self.state.E[s]
            self.state.Ia[s] = r
        for s, r in Is_transitions.items():
            del self.state.E[s]
            self.state.Is[s] = r
        
        # This covers spontaneous infections.
        spontaneous = np.random.poisson(self.parameters.daily_spontaneous_prob * 
                                        len(self.state.S))
        if spontaneous:
            to_infect = set(np.random.choice(list(self.state.S), 
                                             replace = True, 
                                             size = spontaneous))
            for student in to_infect:
                self.infect(student, self.date)
    
    def infection_transmissions(self, verbose = True):
        date = self.date
        spreaders = list(self.state.Ia) + list(self.state.Is)
        if len(spreaders):
            exposed_courses = defaultdict(list)
            for s in spreaders:
                for c in self.semester.student_enrollment[s]:
                    if c in self.semester.meeting_dates[date]:
                        exposed_courses[c].append(s)

            for course, infectious in exposed_courses.items():
                if not len(self.state.Sc[course]):
                    continue
                
                students = list(self.state.Sc[course])
                preclass_time = self.parameters.preclass_interaction_time if \
                        self.semester.meeting_type[course] == MeetType.COURSE else 0
                if course not in self.semester.meeting_rooms:
                    weight = len(infectious) * \
                            (preclass_time +  self.semester.meeting_dates[date][course]) * \
                            self.parameters.rate
                    if weight <= 1: # if it's cheaper to infect n people
                        number_to_infect = np.random.poisson(weight * len(self.state.Sc[course]))
                        to_infect = set(np.random.choice(list(self.state.Sc[course]),
                                                         size = number_to_infect))
                    else: # if it's cheaper to flip coins for each individual
                        p = 1 - np.exp(-weight)
                        infection_vector = np.random.binomial(1, p, size = 
                                                              len(self.state.Sc[course]))
                        to_infect = set()
                        for peer, infect in zip(self.state.Sc[course], infection_vector):
                            if infect:
                                to_infect.add(peer)
                else: # if it is in self.semester.meeting_rooms
                    weights = np.zeros(shape = len(students), dtype = float)
                    for i in infectious:
                        for j, s in enumerate(students):
                            weights[j] += self.semester.meeting_rooms[course][i][s]
                    weights = preclass_time*len(infectious) +\
                               self.semester.meeting_dates[date][course] * weights
                    weights *= self.parameters.rate
                    if weights.sum() <= len(students): # if it's cheaper to infect n people
                        number_to_infect = np.random.poisson(weights.sum())
                        if number_to_infect == 0:
                            continue
                        to_infect = set(np.random.choice(students, 
                                                         size = number_to_infect,
                                                         p = weights / weights.sum()))
                    else: # if it's cheaper to flip coins for each individual
                        p = 1 - np.exp(-weights) # vector of probabilities
                        infection_vector = np.random.binomial(1, p)
                        to_infect = set()
                        for peer, infect in zip(self.state.Sc[course], infection_vector):
                            if infect:
                                to_infect.add(peer)
                for peer in to_infect:
                    if peer in self.state.S:
                        self.infect(peer, date)
        
        
    
    def infect(self, student, date):
        ''' Label a person as infected as of a certain date. '''
        if student in self.state.S:
            for c in self.semester.student_enrollment[student]:
                self.state.Sc[c].remove(student)
            infec_d, remove_d, symptomatic = self.parameters.infection_duration.duration(
                    self, student, date)
            self.state.S.remove(student)
            self.state.E[student] = (infec_d, remove_d, symptomatic)
    
    def initiate_quarantine(self, student, date, duration = None, positive_test = False):
        if duration == None:
            duration = self.parameters.quarantine_length
        if student in self.state.Is:
            if positive_test:
                self.state.R[student] = date
            else:
                end_date = min(date + timedelta(days = duration), self.state.Is[student])
                self.state.Qs[student] = end_date
            del self.state.Is[student]
        elif student in self.state.Ia:
            if positive_test:
                self.state.R[student] = date
            else:
                self.state.Qa[student] = date + timedelta(days = duration)
            del self.state.Ia[student]
        elif student in self.state.E:
            if positive_test:
                self.state.R[student] = date
            else:
                self.state.Qe[student] = self.state.E[student]
            del self.state.E[student]
        elif student in self.state.S:
            self.state.S.remove(student)
            self.state.Q[student] = date + timedelta(days = duration)
            for c in self.semester.student_enrollment[student]:
                self.state.Sc[c].remove(student)
    
    def schedule_positive_test_result(self, student):
        self.positive_tests[self.date + timedelta(days = 1)].add(student)
    
    def request_testing(self, student):
        self.test_requests[self.date].add(student)
    
    def update_statistics(self, verbose = False):
        self.statistics.Tn.append(self.date)
        self.statistics.Qn.append(len(self.state.Q) + len(self.state.Qe) + len(self.state.Qs) + len(self.state.Qa))
        self.statistics.Sn.append(len(self.state.S) + len(self.state.Q))
        self.statistics.En.append(len(self.semester.students) - self.statistics.Sn[-1])
        self.statistics.In.append(len(self.state.Ia) + len(self.state.Is))
        self.statistics.Rn.append(len(self.state.R))

        if verbose:
            print(str(self.date).split(' ')[0], 
                  'Susceptible:', self.statistics.Sn[-1], 
                  'Exposed:', self.statistics.En[-1], 
                  'Infectious:', self.statistics.In[-1], 
                  'Q:', self.statistics.Qn[-1],
                  'Removed:', self.statistics.Rn[-1])
    
    def get_statistics(self):
        return self.statistics.Tn, self.statistics.Sn, self.statistics.En, \
                self.statistics.In, self.statistics.Rn, self.statistics.Qn

    def run(self, verbose = False):
        while self.date <= self.parameters.end_date:
            self.daily_update()
            self.parameters.contact_tracing.trace(self, self.date)
            self.parameters.intervention_policy.testing(self, self.date)
            self.infection_transmissions()
            self.update_statistics(verbose = verbose)
            self.date += timedelta(days = 1)
    
# Convenience function for running many repetitions of an experiment.

def run_repetitions(semester, parameters, save = None):
    # Code to run the simulations...
    print('Running', parameters.repetitions, 'repetitions.')
    start_time = datetime.now()
    Ss = defaultdict(list)
    Es = defaultdict(list)
    Is = defaultdict(list)
    Rs = defaultdict(list)
    Qs = defaultdict(list)
    if parameters.verbose:
        print('Repetition:', end =" ")
    for i in range(parameters.repetitions):
        if parameters.verbose:
            print(i + 1, end =" ")
        
        this_semester = parameters.preprocess(semester)
        simulation = Simulation(this_semester, parameters)
        if type(parameters.initial_exposure) == int:
            parameters.initial_exposure = np.random.choice(list(this_semester.students), 
                    replace = False, size = parameters.initial_exposure)
        for s in parameters.initial_exposure:
            simulation.infect(s, parameters.start_date)
        
        simulation.run(verbose = parameters.repetitions == 1)
        T, S, E, I, R, Q = simulation.get_statistics()
        for t, s, e, i, r, q in zip(T, S, E, I, R, Q):
            Ss[t].append(s)
            Es[t].append(e)
            Is[t].append(i)
            Rs[t].append(r)
            Qs[t].append(q)

    # Print text summary of the results.
    if parameters.verbose:
        print()
    print('Initial exposed:', len(parameters.initial_exposure))
    print('Average final exposure count is', np.mean(Rs[max(Rs)]))
    if (len(Rs[max(Rs)]) > 1):
        print('Sample Standard deviation is:', np.std(Rs[max(Rs)], ddof = 1))
    comp_time = datetime.now() - start_time
    if parameters.verbose:
        print('Total compuation time:', comp_time)
        print('Average computation time:', comp_time / parameters.repetitions)
    
    quarantines = np.array([np.mean(Qs[t]) for t in T])
    print('Quarantine: max', max(quarantines), ', avg', np.mean(quarantines))

    # Plot mean values.
    locator = mdates.AutoDateLocator(interval_multiples = False)
    formatter = mdates.AutoDateFormatter(locator)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16, 4.5))
    ax1.plot(T, [np.mean(Ss[t]) for t in T])
    ax1.plot(T, [np.mean(Es[t]) for t in T])
    ax1.plot(T, [np.mean(Is[t]) for t in T])
    ax1.plot(T, [np.mean(Rs[t]) for t in T])
    ax1.plot(T, [np.mean(Qs[t]) for t in T])
    ax1.legend(['S', 'E', 'I', 'R', 'Q'])
    ax1.set_title('Epidemic Progression')
    ax1.set_xlabel('Day of Year')
    ax1.set_ylabel('Number of Students')
    ax1.xaxis.set_major_locator(locator) # Annoying code to make dates work!
    ax1.xaxis.set_major_formatter(formatter)
    
    # Plot percentiles for total exposed.
    ps = [1, 5, 25, 50, 75, 95, 100]
    ps.sort(reverse = True)
    Ep = {p : [np.percentile(Es[t], p) for t in T] for p in ps}
    for p in ps:
        ax2.plot(T, Ep[p])
    ax2.legend(list(map(lambda x : str(x) + '%', ps)))
    ax2.set_title('Percentiles of Exposed Students')
    ax2.set_xlabel('Day of Year')
    ax2.set_ylabel('Number of Students')
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)

    # Plot percentiles for number infectious.
    Ip = {p : [np.percentile(Is[t], p) for t in T] for p in ps}
    for p in ps:
        ax3.plot(T, Ip[p])
    ax3.legend(list(map(lambda x : str(x) + '%', ps)))
    ax3.set_title('Percentiles of Infectious Students')
    ax3.set_xlabel('Day of Year')
    ax3.set_ylabel('Number of Students')
    ax3.xaxis.set_major_locator(locator)
    ax3.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()

# Additional support.

class IpGeneralTesting:
    def __init__(self, test_groups, test_percentage = 1.0):
        self.test_groups = test_groups
        self.test_percentage = test_percentage
    def testing(self, simulation, date):
        to_test = set(self.test_groups[date.weekday()])
        to_test.update(simulation.test_requests[date])
        for s in to_test:
            if s in simulation.state.Ia or s in simulation.state.Is or \
                    s in simulation.state.Qa or s in simulation.state.Qs:
                simulation.schedule_positive_test_result(s)

def IpWeeklyTesting(semester, weekday = 0):
    test_groups = defaultdict(set)
    weekday = weekday % 7
    for s in semester.students:
        test_groups[weekday].add(s)
    return IpGeneralTesting(test_groups)

def IpWeekdayTesting(semester):
    test_groups = defaultdict(set)
    for i, s in enumerate(np.random.permutation(list(semester.students))):
        test_groups[i % 5].add(s)
    return IpGeneralTesting(test_groups)

def IpDailyTesting(semester):
    test_groups = defaultdict(set)
    for i, s in enumerate(np.random.permutation(list(semester.students))):
        test_groups[i % 7].add(s)
    return IpGeneralTesting(test_groups)

def IpMondayTesting(semester):
    return IpWeeklyTesting(semester, weekday = 0)

def IpThursdayTesting(semester):
    return IpWeeklyTesting(semester, weekday = 3)

class IpRollingTesting:
    def __init__(self, semester, days = 5):
        self.days = 5
        self.test_groups = defaultdict(set)
        for i, s in enumerate(semester.students):
            self.test_groups[i % days].add(s)
    def testing(self, simulation, date):
        group = date.timetuple().tm_yday % self.days
        to_test = set(self.test_groups[group])
        to_test.update(simulation.test_requests[date])
        tomorrow = date + timedelta(days = 1)
        for s in to_test:
            if s in simulation.Ia or s in simulation.Is or \
                    s in simulation.Qa or s in simulation.Qs:
                simulation.positive_tests[tomorrow].add(s)

def make_alternate_hybrid(semester_base):
    ''' Hybrid Schedule: 50% of classes online one week, others online the other weeks. '''
    semester = copy.deepcopy(semester_base)
    meeting_count = {m : i for i, m in enumerate(semester.meetings)}
    date = min(semester.meeting_dates)
    end_date = max(semester.meeting_dates)
    while date <= end_date:
        for i in range(7):
            today = date + timedelta(days = i)
            original = semester.meeting_dates[today]
            new_schedule = {m : w for m, w in original.items() 
                            if meeting_count[m] % 2 or 
                            semester.meeting_type[m] != MeetType.COURSE}
            semester.meeting_dates[today] = new_schedule
        for i in range(7):
            today = date + timedelta(days = i)
            original = semester.meeting_dates[today]
            new_schedule = {m : w for m, w in original.items() 
                            if meeting_count[m] % 2 == 1 or
                            semester.meeting_type[m] != MeetType.COURSE}
            semester.meeting_dates[today] = new_schedule
        date += timedelta(days = 14)
    return semester

def make_alternate_splitclass(semester_base):
    ''' Make a new semester by splitting courses into two sections, one per week. '''
    semester = copy.deepcopy(semester_base)
    new_student_enrollment = defaultdict(set)
    new_meeting_enrollment = defaultdict(set)
    new_meeting_dates = defaultdict(dict)
    new_meetings = {}
    violations = 0
    
    for meet, students in semester.meeting_enrollment.items():
        if semester.meeting_type[meet] != MeetType.COURSE:
            new_meetings[meet] = semester.meetings[meet]
            new_meeting_enrollment[meet] = set(students)
            for s in students:
                new_student_enrollment[s].add(meet)
            date = min(semester.meeting_dates)
            end_date = max(semester.meeting_dates)
            while date <= end_date:
                if meet in semester.meeting_dates[date]:
                    new_meeting_dates[date][meet] = semester.meeting_dates[date][meet]
                date += timedelta(days = 1)
        else:
            violations += len(students) == 1
            week_one_students = set()
            week_two_students = set()
            for i, student in enumerate(students):
                if i % 2:
                    week_two_students.add(student)
                else:
                    week_one_students.add(student)
            for i, ss in enumerate([week_one_students, week_two_students]):
                if len(ss):
                    name = meet + ' W' + str(i + 1)
                    new_meetings[name] = semester.meetings[meet]
                    new_meeting_enrollment[name] = ss
                    for s in ss:
                        new_student_enrollment[s].add(name)
                    date = min(semester.meeting_dates)
                    end_date = max(semester.meeting_dates)
                    week_count = (i + 1) % 2
                    day_count = 0
                    while date <= end_date:
                        if week_count and meet in semester.meeting_dates[date]:
                            new_meeting_dates[date][name] = semester.meeting_dates[date][meet]
                        date += timedelta(days = 1)
                        day_count += 1
                        if day_count >= 7:
                            week_count = (week_count + 1) % 2
                            day_count = 0
    
    semester.meetings = new_meetings
    semester.student_enrollment = new_student_enrollment
    semester.meeting_enrollment = new_meeting_enrollment
    semester.meeting_dates = new_meeting_dates
    return semester

def make_alternate_smallclasses(semester_base, max_size = 50):
    ''' No large course: only classes with at most 50 students. '''
    semester = copy.deepcopy(semester_base)
    canceled_meetings = [m for m, ss in semester.meeting_enrollment.items() 
                        if len(ss) > max_size 
                        and semester.meeting_type[m] == MeetType.COURSE]
    for m in canceled_meetings:
        semester.remove_meeting(m)
    return semester

# Tools for making social groups.

class Cluster:
    def __init__(self, members, meetings):
        self.members = members
        self.meetings = meetings

class ClusterSettings:
    ''' Settings class to give to a randomized cluster generator. '''
    def __init__(self, start_date, end_date, 
                 weekday_group_count, weekday_group_size, weekday_group_time,
                 weekend_group_count, weekend_group_size, weekend_group_time):
        self.start_date = start_date
        self.end_date = end_date
        self.weekend_group_count = weekend_group_count
        self.weekend_group_size = weekend_group_size
        self.weekend_group_time = weekend_group_time
        self.weekday_group_count = weekday_group_count
        self.weekday_group_size = weekday_group_size
        self.weekday_group_time = weekday_group_time


def make_randomized_cluster(student_set, cluster_settings, date):
    if date is not None and date.weekday() in [5, 6]:
        group_count = cluster_settings.weekend_group_count
        group_size = cluster_settings.weekend_group_size
        group_time = cluster_settings.weekend_group_time
    else:
        group_count = cluster_settings.weekday_group_count
        group_size = cluster_settings.weekday_group_size
        group_time = cluster_settings.weekday_group_time
    group_count = min(group_count, len(student_set) // group_size)

    groups = np.random.choice(list(student_set), 
                              replace = False, 
                              size = group_size * group_count)
    clusters = []
    for g in range(group_count):
        group = groups[group_size * g: group_size * (g + 1)]
        if date is None:
            meet = {}
            date = cluster_settings.start_date
            while date <= cluster_settings.end_date:
                meet[date] = group_time
                date += timedelta(days = 1)
        else:
            meet = {date : group_time}
        cluster = Cluster(group, meet)
        clusters.append(cluster)
    return clusters

def make_randomized_clusters(student_set, cluster_settings):
    clusters = []
    date = cluster_settings.start_date
    while date <= cluster_settings.end_date:
        clusters.extend(make_randomized_cluster(student_set, cluster_settings, date))
        date += timedelta(days = 1)
    
    return clusters

def make_randomized_static_clusters(student_set, cluster_settings):
    return make_randomized_cluster(student_set, cluster_settings, None)

def make_from_clusters(semester_base, clusters, prefix = ''):
    ''' Given a semester, a set of clusters, apply the clusters to the semester. '''
    semester = copy.deepcopy(semester_base)
    for i, cluster in enumerate(clusters):
        semester.add_meeting('AutoCluster ' + prefix + str(i),
                             {},
                             cluster.meetings,
                             cluster.members,
                             meet_type = MeetType.SOCIAL)
    
    return semester

def make_social_groups_pairs(semester, fraction_paired, interaction_time = 1200,
                             excluded = set(), weighted = True):
    students = [x for x in semester.students if x not in excluded]
    def all_days():
        date = min(semester.meeting_dates)
        end_date = max(semester.meeting_dates)
        while date <= end_date:
            yield date
            date += timedelta(days = 1)
    
    generic_meeting_time = {d : interaction_time for d in all_days()}
    
    pool_size = int(len(students) * fraction_paired)
    pair_pool = np.random.choice(students, size = pool_size)
    pair_pool_remaining = set(pair_pool)
    new_courses = {}
    def match_quality(s1, s2):
        mq = 1.0
        for course in semester.student_enrollment[s1]:
            mq += s2 in semester.meeting_enrollment[course]
        return mq
    
    clusters = []
    if weighted:
        for s in pair_pool:
            if s not in pair_pool_remaining:
                continue
            if len(pair_pool_remaining) < 2:
                break
            pair_pool_remaining.remove(s)
            matches = list(pair_pool_remaining)
            probability = np.array([match_quality(s, s2) for s2 in matches])
            probability /= sum(probability)
            m = np.random.choice(matches, p = probability)
            pair_pool_remaining.remove(m)
            clusters.append(Cluster({s, m}, generic_meeting_time))
    else:
        np.random.shuffle(pair_pool)
        i = 0
        while i + 1 < len(pair_pool):
            clusters.append(Cluster({pair_pool[i], pair_pool[i + 1]}, generic_meeting_time))
            i += 2
    
    return clusters, set(semester.students)

def make_social_groups_varsity(semester, 
                               weekday_size, weekday_time, 
                               weekend_size, weekend_time):
    varsity = {name : members for name, members in semester.groups.items()
               if 'Varsity' in name}
    max_size = max([len(x) for x in varsity.values()])
    cluster_settings = ClusterSettings(min(semester.meeting_dates), max(semester.meeting_dates),
                                       max_size, weekday_size, weekday_time,
                                       max_size, weekend_size, weekend_time)
    clusters = []
    processed_students = set()
    for sport, roster in varsity.items():
        clusters.extend(make_randomized_clusters(roster, cluster_settings))
        processed_students.update(roster)
    
    return clusters, processed_students

def make_social_groups_varsity_static(semester, size, time):
    varsity = {name : members for name, members in semester.groups.items()
               if 'Varsity' in name}
    max_size = max([len(x) for x in varsity.values()])
    cluster_settings = ClusterSettings(min(semester.meeting_dates), max(semester.meeting_dates),
                                       max_size, size, time,
                                       max_size, size, time)
    clusters = []
    processed_students = set()
    for sport, roster in varsity.items():
        clusters.extend(make_randomized_static_clusters(roster, cluster_settings))
        processed_students.update(roster)
    
    return clusters, processed_students

class BasicContactTracing:
    def __init__(self, quarantine_length, trace_length = 3):
        self.quarantine_length = quarantine_length
        self.trace_length = trace_length
    def trace(self, simulation, date):
        ''' For each student to trace, find all of their peers from the past
            3 days.  Then, quarantine all those peers for some days and
            schedule a test for them. '''
        trace_dates = [date - timedelta(days = d + 1) for d in range(self.trace_length)]
        for s in simulation.contact_trace_request[date]:
            recent_meets = set()
            for m in simulation.semester.student_enrollment[s]:
                if simulation.semester.meeting_type[m] != MeetType.UNTRACEABLE:
                    for d in trace_dates:
                        if m in simulation.semester.meeting_dates[d]:
                            recent_meets.add(m)
                            break
            peers = set()
            for m in recent_meets:
                peers.update(simulation.semester.meeting_enrollment[m])
            if len(peers):
                peers.remove(s)
                for peer in peers:
                    simulation.initiate_quarantine(peer, date, self.quarantine_length)

