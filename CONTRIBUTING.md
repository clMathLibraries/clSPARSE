## Contributor guidelines

Contributing code to this project is intended to be light weight and intuitive to users familiar with GitHub to actively encourage contributions.  

## The mechanics of contributing code
Firstly, in order to contribute code to this project, a contributor must have a valid and current [GitHub account](https://help.github.com/articles/set-up-git) available to use.  Given an account,
* The potential contributor forks this project into his/her account following the traditional [forking](https://help.github.com/articles/fork-a-repo) model native to GitHub
* After forking, the contributor [clones their repository](https://help.github.com/articles/create-a-repo) locally on their machine
* Code is developed and checked into the contributor's repository.  These commits are eventually pushed upstream to their GitHub repository
* The contributor then issues a [pull-request](https://help.github.com/articles/using-pull-requests) against the **develop** branch of this repository, which is the [git flow](http://nvie.com/posts/a-successful-git-branching-model/) workflow which is well suited for working with GitHub
    * A [git extention](https://github.com/nvie/gitflow) has been developed to ease the use of the 'git flow' methodology, but requires manual installation by the user.  Refer to the projects wiki

At this point, the repository maintainers will be notified by GitHub that a 'pull request' exists pending against their repository.  A code review should be completed within a few days, depending on the scope of submitted code, and the code will either be accepted, rejected or commented on for feedback.

## Code submission guidelines
We want to ensure that the project code base maintains a level of quality over time, such that future contributors find it as easy to jump into the code as hopefully it is today.  As such, pull requests should
* follow the [code style guidelines]( ) of the project as posted to the project wiki.  Unfortunately, there was no unifying code guidelines defined between the BLAS & FFT projects, but code submissions should not mix styles within an individual file.  We have since defined and posted a code style guideline for the projects and we expect the code to slowly transition to the new
guidelines over time
    *  separate check-ins that modify a files style from the ones that add/change/delete code.
* target the **develop** branch
* ensure that the [code properly builds]( https://github.com/kknox/clSPARSE/wiki/Build )
  * Proper cloud based build services should verify the code builds in a PR
* cannot break existing test cases
    * we encourage contributors to [run all tests]( https://github.com/kknox/clSPARSE/wiki/Testing ) before issuing a pull-request
        * if possible, upload the test results associated with the pull request to a personal [gist repository]( https://gist.github.com/ ) and insert a link to the test results in the pull request so that collaborators can browse the results
        * if no test results are provided with the pull request, official collaborators will run the test suite on their test machines against the patch before we will accept the pull-request
            * if we detect failing test cases, we will request that the code associated with the pull request be fixed before the pull request will be merged
    * if new functionality is introduced with the pull request, sufficient test cases should be added to verify the new functionality is correct
        * new tests should integrate with the existing [googletest framework]( https://code.google.com/p/googletest/wiki/Primer ) located in the src/tests directory of the repo
        * if collaborators feel the new tests do not provide sufficient coverage, feedback on the pull request will be left with suggestions on how to improve the tests before the pull request will be merged

Pull requests will be reviewed by the set of collaborators that are assigned for the repository.  Pull requests may be accepted, declined or a conversation may start on the pull request thread with feedback.  At any time, collaborators may decline a pull request if they decide the contribution is not appropriate for the project, or the feedback from reviewers on a pull request is not being addressed in an appropriate amount of time.

## Is it possible to become an official collaborator of the repository?
Yes, we hope to promote trusted members of the community, who prove themselves competent and request to take on the extra responsibility to be official collaborators of the project.  It is worth noting, that on GitHub everybody has read-only access to the source and that everybody has the ability to issue a pull request to contribute to the project.  The benefit of being a repository collaborator allows you to be able to be able to manage other peoples pull requests.
